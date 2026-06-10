#!/usr/bin/env node

/**
 * Dashboard build CLI — parses args and invokes the build orchestrator.
 *
 * Usage:
 *   node dashboard/build.mjs --wikiRoot=<path> --outDir=<path>
 *
 * Defaults:
 *   --wikiRoot = cwd/wiki
 *   --outDir = cwd/dashboard
 */

import { fileURLToPath } from 'node:url';
import { dirname, resolve, join } from 'node:path';
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync, mkdirSync } from 'node:fs';

// Import renderers from plugin
const pluginRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../jaewon-plugin-learning-dev/jaewon-plugin-learning/dashboard');

function importFromPlugin(moduleName) {
  return import(`file://${join(pluginRoot, moduleName)}`);
}

// Parse CLI arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const cwd = process.cwd();
  const opts = {
    wikiRoot: join(cwd, 'wiki'),
    outDir: join(cwd, 'dashboard'),
  };

  for (const arg of args) {
    if (arg.startsWith('--wikiRoot=')) {
      opts.wikiRoot = arg.slice('--wikiRoot='.length);
    } else if (arg.startsWith('--outDir=')) {
      opts.outDir = arg.slice('--outDir='.length);
    }
  }

  return opts;
}

// Wiki readers (duplicated from plugin build.mjs for independence)
function readJsonFile(filePath) {
  try {
    return JSON.parse(readFileSync(filePath, 'utf-8'));
  } catch {
    return null;
  }
}

function readTextFile(filePath) {
  try {
    return readFileSync(filePath, 'utf-8');
  } catch {
    return '';
  }
}

function scanCourses(wikiRoot) {
  const coursesDir = join(wikiRoot, 'courses');
  if (!existsSync(coursesDir)) return [];

  const slugs = readdirSync(coursesDir).filter((name) => {
    return statSync(join(coursesDir, name)).isDirectory();
  });

  return slugs.map((slug) => {
    const courseDir = join(coursesDir, slug);
    const meta = readJsonFile(join(courseDir, 'meta.json')) || { slug, title: slug };

    const chapterDirs = readdirSync(courseDir).filter((name) => {
      const full = join(courseDir, name);
      return statSync(full).isDirectory();
    }).sort();

    const chapters = chapterDirs.map((chSlug) => {
      const verdict = readJsonFile(join(courseDir, chSlug, 'verdict.json'));
      return { slug: chSlug, title: chSlug, verdict };
    });

    return { ...meta, slug, chapters };
  });
}

function readProfile(wikiRoot) {
  const learnerDir = join(wikiRoot, 'learner');
  return {
    style:        readTextFile(join(learnerDir, 'learning-style.md')),
    strengths:    readTextFile(join(learnerDir, 'strengths.md')),
    weaknesses:   readTextFile(join(learnerDir, 'weaknesses.md')),
    push_tactics: readTextFile(join(learnerDir, 'push-tactics.md')),
    session_log:  readTextFile(join(learnerDir, 'session-log.md')),
  };
}

function parseSessionLog(md) {
  const sessions = [];
  const blocks = md.split(/^##\s+/m).slice(1);

  for (const block of blocks) {
    const lines = block.split('\n');
    const date = (lines[0] || '').trim();
    const entry = { date, course: '', chapter: '', phase: '', verdict: null, notes: '' };
    const noteLines = [];

    for (const line of lines.slice(1)) {
      const m = line.match(/^-\s+(\w[\w\s-]*):\s*(.+)/);
      if (m) {
        const key = m[1].trim().toLowerCase();
        const val = m[2].trim();
        if (key === 'course')  entry.course  = val;
        if (key === 'chapter') entry.chapter = val;
        if (key === 'phase')   entry.phase   = val;
        if (key === 'verdict') entry.verdict = val;
        if (key === 'notes')   noteLines.push(val);
      } else if (line.trim()) {
        noteLines.push(line.trim());
      }
    }
    entry.notes = noteLines.join(' ');
    sessions.push(entry);
  }

  return sessions;
}

function writeHtml(outDir, filename, html) {
  const filePath = join(outDir, filename);
  writeFileSync(filePath, html, 'utf-8');
  return filePath;
}

function buildSections({ courses, profile, sessions, verdicts, renderers }) {
  const { renderHome, renderCourse, renderProfile, renderTimeline } = renderers;
  return [
    { out: 'index.html',    render: () => renderHome({ courses, profile }) },
    { out: 'profile.html',  render: () => renderProfile(profile) },
    { out: 'timeline.html', render: () => renderTimeline({ sessions, verdicts }) },
  ];
}

// Main build function
async function build({ wikiRoot, outDir }) {
  const start = Date.now();

  if (!existsSync(wikiRoot)) {
    const err = new Error(`wikiRoot does not exist: ${wikiRoot}`);
    err.code = 'ENOENT';
    throw err;
  }

  mkdirSync(outDir, { recursive: true });

  // Load renderers from plugin
  const { renderHome } = await importFromPlugin('render-home.mjs');
  const { renderCourse } = await importFromPlugin('render-course.mjs');
  const { renderProfile } = await importFromPlugin('render-profile.mjs');
  const { renderTimeline } = await importFromPlugin('render-timeline.mjs');

  const courses  = scanCourses(wikiRoot);
  const profile  = readProfile(wikiRoot);
  const sessions = parseSessionLog(profile.session_log || '');
  const verdicts = courses.flatMap((c) =>
    c.chapters.filter((ch) => ch.verdict).map((ch) => ch.verdict)
  );

  const pagesWritten = [];

  for (const section of buildSections({
    courses,
    profile,
    sessions,
    verdicts,
    renderers: { renderHome, renderCourse, renderProfile, renderTimeline }
  })) {
    pagesWritten.push(writeHtml(outDir, section.out, section.render()));
  }

  for (const course of courses) {
    pagesWritten.push(writeHtml(outDir, `${course.slug}.html`, renderCourse({ course })));
  }

  return { durationMs: Date.now() - start, pagesWritten };
}

// CLI entry point
async function main() {
  const opts = parseArgs();

  try {
    const report = await build(opts);
    console.log(`\nDashboard build complete (${report.durationMs}ms)`);
    console.log(`\nFiles written:`);
    for (const file of report.pagesWritten) {
      console.log(`  - ${file}`);
    }
    process.exit(0);
  } catch (err) {
    console.error(`\nBuild failed:`, err.message);
    process.exit(1);
  }
}

main();
