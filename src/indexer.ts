import fs from 'fs';
import path from 'path';
import os from 'os';
import { initDatabase, insertExchange } from './db.js';
import { parseConversation } from './parser.js';
import { initEmbeddings, generateExchangeEmbedding } from './embeddings.js';
import { summarizeConversation } from './summarizer.js';
import { ConversationExchange } from './types.js';
import { getArchiveDir, getExcludeConfigPath } from './paths.js';

// Set max output tokens for Claude SDK (used by summarizer)
process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = '20000';

// Markers that indicate a conversation should not be indexed
// (e.g., summarization sessions, internal tooling)
const EXCLUSION_MARKERS = [
  '<INSTRUCTIONS-TO-EPISODIC-MEMORY>DO NOT INDEX THIS CHAT</INSTRUCTIONS-TO-EPISODIC-MEMORY>',
  'Only use NO_INSIGHTS_FOUND',
  'Context: This summary will be shown in a list to help users and Claude choose which conversations are relevant',
];

// Minimum number of actual messages (user/assistant) for a conversation to be worth indexing
const MIN_MESSAGE_COUNT = 5;

function shouldSkipConversation(filePath: string): boolean {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');

    // Check for exclusion markers
    if (EXCLUSION_MARKERS.some(marker => content.includes(marker))) {
      return true;
    }

    // Count actual user/assistant messages (not queue-operations, summaries, etc.)
    const lines = content.split('\n').filter(line => line.trim());
    let messageCount = 0;
    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        if (entry.type === 'user' || entry.type === 'assistant') {
          messageCount++;
        }
      } catch {
        // Skip malformed lines
      }
    }

    // Skip conversations with too few messages
    if (messageCount < MIN_MESSAGE_COUNT) {
      return true;
    }

    return false;
  } catch (error) {
    return false;
  }
}

// Increase max listeners for concurrent API calls
import { EventEmitter } from 'events';
EventEmitter.defaultMaxListeners = 20;

// Allow overriding paths for testing
function getProjectsDir(): string {
  return process.env.TEST_PROJECTS_DIR || path.join(os.homedir(), '.claude', 'projects');
}

// Projects to exclude from indexing (configurable via env or config file)
function getExcludedProjects(): string[] {
  // Check env variable first
  if (process.env.CONVERSATION_SEARCH_EXCLUDE_PROJECTS) {
    return process.env.CONVERSATION_SEARCH_EXCLUDE_PROJECTS.split(',').map(p => p.trim());
  }

  // Check for config file
  const configPath = getExcludeConfigPath();
  if (fs.existsSync(configPath)) {
    const content = fs.readFileSync(configPath, 'utf-8');
    return content.split('\n').map(line => line.trim()).filter(line => line && !line.startsWith('#'));
  }

  // Default: no exclusions
  return [];
}

// Process items in batches with limited concurrency
async function processBatch<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  concurrency: number
): Promise<R[]> {
  const results: R[] = [];

  for (let i = 0; i < items.length; i += concurrency) {
    const batch = items.slice(i, i + concurrency);
    const batchResults = await Promise.all(batch.map(processor));
    results.push(...batchResults);
  }

  return results;
}

export async function indexConversations(
  limitToProject?: string,
  maxConversations?: number,
  concurrency: number = 1,
  noSummaries: boolean = false
): Promise<void> {
  console.log('Initializing database...');
  const db = initDatabase();

  console.log('Loading embedding model...');
  await initEmbeddings();

  if (noSummaries) {
    console.log('⚠️  Running in no-summaries mode (skipping AI summaries)');
  }

  console.log('Scanning for conversation files...');
  const PROJECTS_DIR = getProjectsDir();
  const ARCHIVE_DIR = getArchiveDir(); // Now uses paths.ts
  const projects = fs.readdirSync(PROJECTS_DIR);

  let totalExchanges = 0;
  let conversationsProcessed = 0;

  const excludedProjects = getExcludedProjects();

  for (const project of projects) {
    // Skip excluded projects
    if (excludedProjects.includes(project)) {
      console.log(`\nSkipping excluded project: ${project}`);
      continue;
    }

    // Skip if limiting to specific project
    if (limitToProject && project !== limitToProject) continue;
    const projectPath = path.join(PROJECTS_DIR, project);
    const stat = fs.statSync(projectPath);

    if (!stat.isDirectory()) continue;

    const files = fs.readdirSync(projectPath).filter(f => f.endsWith('.jsonl'));

    if (files.length === 0) continue;

    console.log(`\nProcessing project: ${project} (${files.length} conversations)`);
    if (concurrency > 1) console.log(`  Concurrency: ${concurrency}`);

    // Create archive directory for this project
    const projectArchive = path.join(ARCHIVE_DIR, project);
    fs.mkdirSync(projectArchive, { recursive: true });

    // Prepare all conversations first
    type ConvToProcess = {
      file: string;
      sourcePath: string;
      archivePath: string;
      summaryPath: string;
      exchanges: ConversationExchange[];
    };

    const toProcess: ConvToProcess[] = [];

    for (const file of files) {
      try {
        const sourcePath = path.join(projectPath, file);
        const archivePath = path.join(projectArchive, file);

        // Skip conversations with exclusion markers (e.g., summarization sessions)
        if (shouldSkipConversation(sourcePath)) {
          console.log(`  Skipped ${file} (excluded)`);
          continue;
        }

        // Copy to archive
        if (!fs.existsSync(archivePath)) {
          fs.copyFileSync(sourcePath, archivePath);
          console.log(`  Archived: ${file}`);
        }

        // Parse conversation
        const exchanges = await parseConversation(sourcePath, project, archivePath);

        if (exchanges.length === 0) {
          console.log(`  Skipped ${file} (no exchanges)`);
          continue;
        }

        toProcess.push({
          file,
          sourcePath,
          archivePath,
          summaryPath: archivePath.replace('.jsonl', '-summary.txt'),
          exchanges
        });
      } catch (error) {
        // Log error but continue processing other files
        console.error(`  Error processing ${file}: ${error instanceof Error ? error.message : error}`);
      }
    }

    // Batch summarize conversations in parallel (unless --no-summaries)
    if (!noSummaries) {
      const needsSummary = toProcess.filter(c => !fs.existsSync(c.summaryPath));

      if (needsSummary.length > 0) {
        console.log(`  Generating ${needsSummary.length} summaries (concurrency: ${concurrency})...`);

        await processBatch(needsSummary, async (conv) => {
          try {
            const summary = await summarizeConversation(conv.exchanges);
            fs.writeFileSync(conv.summaryPath, summary, 'utf-8');
            const wordCount = summary.split(/\s+/).length;
            console.log(`  ✓ ${conv.file}: ${wordCount} words`);
            return summary;
          } catch (error) {
            console.log(`  ✗ ${conv.file}: ${error}`);
            return null;
          }
        }, concurrency);
      }
    } else {
      console.log(`  Skipping ${toProcess.length} summaries (--no-summaries mode)`);
    }

    // Now process embeddings and DB inserts (fast, sequential is fine)
    for (const conv of toProcess) {
      for (const exchange of conv.exchanges) {
        const toolNames = exchange.toolCalls?.map(tc => tc.toolName);
        const embedding = await generateExchangeEmbedding(
          exchange.userMessage,
          exchange.assistantMessage,
          toolNames
        );

        insertExchange(db, exchange, embedding, toolNames);
      }

      totalExchanges += conv.exchanges.length;
      conversationsProcessed++;

      // Check if we hit the limit
      if (maxConversations && conversationsProcessed >= maxConversations) {
        console.log(`\nReached limit of ${maxConversations} conversations`);
        db.close();
        console.log(`✅ Indexing complete! Conversations: ${conversationsProcessed}, Exchanges: ${totalExchanges}`);
        return;
      }
    }
  }

  db.close();
  console.log(`\n✅ Indexing complete! Conversations: ${conversationsProcessed}, Exchanges: ${totalExchanges}`);
}

export async function indexSession(sessionId: string, concurrency: number = 1, noSummaries: boolean = false): Promise<void> {
  console.log(`Indexing session: ${sessionId}`);

  // Find the conversation file for this session
  const PROJECTS_DIR = getProjectsDir();
  const ARCHIVE_DIR = getArchiveDir(); // Now uses paths.ts
  const projects = fs.readdirSync(PROJECTS_DIR);
  const excludedProjects = getExcludedProjects();
  let found = false;

  for (const project of projects) {
    if (excludedProjects.includes(project)) continue;

    const projectPath = path.join(PROJECTS_DIR, project);
    if (!fs.statSync(projectPath).isDirectory()) continue;

    const files = fs.readdirSync(projectPath).filter(f => f.includes(sessionId) && f.endsWith('.jsonl'));

    if (files.length > 0) {
      found = true;
      const file = files[0];
      const sourcePath = path.join(projectPath, file);

      const db = initDatabase();
      await initEmbeddings();

      const projectArchive = path.join(ARCHIVE_DIR, project);
      fs.mkdirSync(projectArchive, { recursive: true });

      const archivePath = path.join(projectArchive, file);

      // Archive
      if (!fs.existsSync(archivePath)) {
        fs.copyFileSync(sourcePath, archivePath);
      }

      // Parse and summarize
      const exchanges = await parseConversation(sourcePath, project, archivePath);

      if (exchanges.length > 0) {
        // Generate summary (unless --no-summaries)
        const summaryPath = archivePath.replace('.jsonl', '-summary.txt');
        if (!noSummaries && !fs.existsSync(summaryPath)) {
          const summary = await summarizeConversation(exchanges);
          fs.writeFileSync(summaryPath, summary, 'utf-8');
          console.log(`Summary: ${summary.split(/\s+/).length} words`);
        }

        // Index
        for (const exchange of exchanges) {
          const toolNames = exchange.toolCalls?.map(tc => tc.toolName);
          const embedding = await generateExchangeEmbedding(
            exchange.userMessage,
            exchange.assistantMessage,
            toolNames
          );
          insertExchange(db, exchange, embedding, toolNames);
        }

        console.log(`✅ Indexed session ${sessionId}: ${exchanges.length} exchanges`);
      }

      db.close();
      break;
    }
  }

  if (!found) {
    console.log(`Session ${sessionId} not found`);
  }
}

export async function indexUnprocessed(concurrency: number = 1, noSummaries: boolean = false): Promise<void> {
  console.log('Finding unprocessed conversations...');
  if (concurrency > 1) console.log(`Concurrency: ${concurrency}`);
  if (noSummaries) console.log('⚠️  Running in no-summaries mode (skipping AI summaries)');

  const db = initDatabase();
  await initEmbeddings();

  const PROJECTS_DIR = getProjectsDir();
  const ARCHIVE_DIR = getArchiveDir(); // Now uses paths.ts
  const projects = fs.readdirSync(PROJECTS_DIR);
  const excludedProjects = getExcludedProjects();

  // Load all indexed paths into memory for O(1) lookups (much faster than per-file queries)
  const indexedPaths = new Set<string>();
  const indexedRows = db.prepare('SELECT DISTINCT archive_path FROM exchanges').all() as Array<{ archive_path: string }>;
  for (const row of indexedRows) {
    indexedPaths.add(row.archive_path);
  }

  type UnprocessedConv = {
    project: string;
    file: string;
    sourcePath: string;
    archivePath: string;
    summaryPath: string;
    exchanges: ConversationExchange[];
  };

  const unprocessed: UnprocessedConv[] = [];

  // Collect all unprocessed conversations
  for (const project of projects) {
    if (excludedProjects.includes(project)) continue;

    const projectPath = path.join(PROJECTS_DIR, project);
    if (!fs.statSync(projectPath).isDirectory()) continue;

    const files = fs.readdirSync(projectPath).filter(f => f.endsWith('.jsonl'));

    for (const file of files) {
      try {
        const sourcePath = path.join(projectPath, file);
        const projectArchive = path.join(ARCHIVE_DIR, project);
        const archivePath = path.join(projectArchive, file);
        const summaryPath = archivePath.replace('.jsonl', '-summary.txt');

        // Check if already indexed (O(1) Set lookup)
        if (indexedPaths.has(archivePath)) continue;

        // Skip conversations with exclusion markers (e.g., summarization sessions)
        if (shouldSkipConversation(sourcePath)) continue;

        fs.mkdirSync(projectArchive, { recursive: true });

        // Archive if needed
        if (!fs.existsSync(archivePath)) {
          fs.copyFileSync(sourcePath, archivePath);
        }

        // Parse and check
        const exchanges = await parseConversation(sourcePath, project, archivePath);
        if (exchanges.length === 0) continue;

        unprocessed.push({ project, file, sourcePath, archivePath, summaryPath, exchanges });
      } catch (error) {
        // Log error but continue processing other files
        console.error(`  Error processing ${project}/${file}: ${error instanceof Error ? error.message : error}`);
      }
    }
  }

  if (unprocessed.length === 0) {
    console.log('✅ All conversations are already processed!');
    db.close();
    return;
  }

  console.log(`Found ${unprocessed.length} unprocessed conversations`);

  // Batch process summaries (unless --no-summaries)
  if (!noSummaries) {
    const needsSummary = unprocessed.filter(c => !fs.existsSync(c.summaryPath));
    if (needsSummary.length > 0) {
      console.log(`Generating ${needsSummary.length} summaries (concurrency: ${concurrency})...\n`);

      await processBatch(needsSummary, async (conv) => {
        try {
          const summary = await summarizeConversation(conv.exchanges);
          fs.writeFileSync(conv.summaryPath, summary, 'utf-8');
          const wordCount = summary.split(/\s+/).length;
          console.log(`  ✓ ${conv.project}/${conv.file}: ${wordCount} words`);
          return summary;
        } catch (error) {
          console.log(`  ✗ ${conv.project}/${conv.file}: ${error}`);
          return null;
        }
      }, concurrency);
    }
  } else {
    console.log(`Skipping summaries for ${unprocessed.length} conversations (--no-summaries mode)\n`);
  }

  // Now index embeddings
  console.log(`\nIndexing embeddings...`);
  for (const conv of unprocessed) {
    for (const exchange of conv.exchanges) {
      const toolNames = exchange.toolCalls?.map(tc => tc.toolName);
      const embedding = await generateExchangeEmbedding(
        exchange.userMessage,
        exchange.assistantMessage,
        toolNames
      );
      insertExchange(db, exchange, embedding, toolNames);
    }
  }

  db.close();
  console.log(`\n✅ Processed ${unprocessed.length} conversations`);
}

export async function pruneShortConversations(dryRun: boolean = true): Promise<{ pruned: number; deleted: string[] }> {
  console.log(dryRun ? 'Scanning for short/excluded conversations (dry run)...' : 'Pruning short/excluded conversations...');

  const db = initDatabase();
  const ARCHIVE_DIR = getArchiveDir();

  const deleted: string[] = [];
  let pruned = 0;

  if (!fs.existsSync(ARCHIVE_DIR)) {
    console.log('No archive directory found.');
    db.close();
    return { pruned: 0, deleted: [] };
  }

  const projects = fs.readdirSync(ARCHIVE_DIR);

  for (const project of projects) {
    const projectPath = path.join(ARCHIVE_DIR, project);
    if (!fs.statSync(projectPath).isDirectory()) continue;

    const files = fs.readdirSync(projectPath).filter(f => f.endsWith('.jsonl'));

    for (const file of files) {
      const archivePath = path.join(projectPath, file);
      const summaryPath = archivePath.replace('.jsonl', '-summary.txt');

      // Check if this file should be skipped (too short or has exclusion markers)
      if (shouldSkipConversation(archivePath)) {
        pruned++;
        deleted.push(archivePath);

        if (!dryRun) {
          // Delete from database
          const rows = db.prepare('SELECT id FROM exchanges WHERE archive_path = ?').all(archivePath) as Array<{ id: string }>;
          for (const row of rows) {
            db.prepare('DELETE FROM vec_exchanges WHERE id = ?').run(row.id);
            db.prepare('DELETE FROM tool_calls WHERE exchange_id = ?').run(row.id);
            db.prepare('DELETE FROM exchanges WHERE id = ?').run(row.id);
          }

          // Delete archive file
          if (fs.existsSync(archivePath)) {
            fs.unlinkSync(archivePath);
          }

          // Delete summary file if exists
          if (fs.existsSync(summaryPath)) {
            fs.unlinkSync(summaryPath);
          }

          console.log(`  Deleted: ${project}/${file}`);
        } else {
          console.log(`  Would delete: ${project}/${file}`);
        }
      }
    }
  }

  db.close();

  if (dryRun) {
    console.log(`\n📋 Found ${pruned} conversations to prune.`);
    console.log('Run with --prune --confirm to delete them.');
  } else {
    console.log(`\n✅ Pruned ${pruned} conversations.`);
  }

  return { pruned, deleted };
}
