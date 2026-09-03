/**
 * Capture bpTorch v0.2.0 UI screenshots for README and social sharing (X/Twitter).
 */
import { chromium } from 'playwright';
import { mkdir, readdir, unlink, stat } from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT = path.dirname(path.dirname(path.dirname(fileURLToPath(import.meta.url))));
const OUT = path.join(ROOT, 'docs/images/x');
const SOCIAL_PREVIEW = path.join(ROOT, 'docs/images/social-preview.jpg');

const EXPECTED_FILES = new Set([
  '01-workspace-overview.png',
  '02-templates-26-samples.png',
  '03-arch26-llama-tiny.png',
  '04-live-train-loss.png',
  '05-playground.png',
  '06-pytorch-code.png',
  '07-parameters-breakdown.png',
  '08-arch4-mlp.png',
  '09-inspector-node.png',
  '10-context-menu.png',
]);

async function openSamplesSubmenu(page) {
  const menu = page.locator('[role="menu"][aria-label="Blueprint templates"]');
  if (!(await menu.count()) || !(await menu.isVisible())) {
    await page.click('button[aria-label="Load blueprint template"]');
    await page.waitForSelector('[role="menu"][aria-label="Blueprint templates"]', { timeout: 8000 });
  }
  const samplesTrigger = page
    .locator('[role="menuitem"]')
    .filter({ hasText: /Architecture Samples/i })
    .first();
  await samplesTrigger.hover();
  await page.waitForTimeout(400);
  await page.waitForSelector('[role="menu"][aria-label="Architecture samples"]', { timeout: 8000 });
}

async function closeDrawerIfOpen(page) {
  const closeBtn = page.locator('button[title="Stop Run"] + button').first();
  if (await closeBtn.count() && await closeBtn.isVisible()) {
    await closeBtn.click();
    await page.waitForTimeout(300);
  }
}

async function loadSample(page, label) {
  const menu = page.locator('[role="menu"][aria-label="Blueprint templates"]');
  if (!(await menu.count()) || !(await menu.isVisible())) {
    await page.click('button[aria-label="Load blueprint template"]');
    await page.waitForSelector('[role="menu"][aria-label="Blueprint templates"]', { timeout: 8000 });
  }
  const samplesTrigger = page
    .locator('[role="menuitem"]')
    .filter({ hasText: /Architecture Samples/i })
    .first();
  await samplesTrigger.hover();
  await page.waitForTimeout(400);
  await page.waitForSelector('[role="menu"][aria-label="Architecture samples"]', { timeout: 8000 });
  
  await page
    .locator('[role="menu"][aria-label="Architecture samples"] [role="menuitem"]')
    .filter({ hasText: label })
    .first()
    .click();

  // Wait 2.5s for load, escape leftover menus, fit view
  await page.waitForTimeout(2500);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);

  const fitViewBtn = page.locator('.react-flow__controls-fitview').first();
  if (await fitViewBtn.count()) {
    await fitViewBtn.click();
    await page.waitForTimeout(500);
  }
}

async function main() {
  await mkdir(OUT, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });

  console.log('Navigating to http://localhost:5173/...');
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
  await page.waitForSelector('button[aria-label="Load blueprint template"]', { timeout: 15000 });
  await page.waitForTimeout(1500);

  // 01-workspace-overview.png
  console.log('Capturing 01-workspace-overview.png...');
  const p01 = path.join(OUT, '01-workspace-overview.png');
  await page.screenshot({ path: p01 });
  console.log('Capturing docs/images/social-preview.jpg...');
  await page.screenshot({ type: 'jpeg', quality: 80, path: SOCIAL_PREVIEW });

  // 02-templates-26-samples.png
  console.log('Capturing 02-templates-26-samples.png...');
  await openSamplesSubmenu(page);
  await page.waitForTimeout(500);
  const p02 = path.join(OUT, '02-templates-26-samples.png');
  await page.screenshot({ path: p02 });
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(300);

  // 03-arch26-llama-tiny.png
  console.log('Capturing 03-arch26-llama-tiny.png...');
  await loadSample(page, 'Arch 26');
  const p03 = path.join(OUT, '03-arch26-llama-tiny.png');
  await page.screenshot({ path: p03 });

  // 04-live-train-loss.png
  console.log('Capturing 04-live-train-loss.png...');
  await loadSample(page, 'Arch 1');
  
  // Set speed to Instant to guarantee rapid training curve points
  const speedSelect = page.locator('select').filter({ hasText: /Speed:/i }).first();
  if (await speedSelect.count()) {
    await speedSelect.selectOption('instant');
  }

  const trainBtn = page.locator('button[aria-label="Train model"]').first();
  if (await trainBtn.count()) {
    await trainBtn.click();
  }

  const lossTab = page.locator('button').filter({ hasText: /Live Loss Plotter/i }).first();
  if (await lossTab.count()) {
    await lossTab.click();
  }
  // Wait 5s for batches to record loss curve points
  await page.waitForTimeout(5000);
  const p04 = path.join(OUT, '04-live-train-loss.png');
  await page.screenshot({ path: p04 });

  // Pause training
  const pauseBtn = page.locator('button[aria-label="Pause training"]').first();
  if (await pauseBtn.count()) {
    await pauseBtn.click();
  }

  // 05-playground.png
  console.log('Capturing 05-playground.png...');
  const playgroundTab = page.locator('button').filter({ hasText: /Playground/i }).first();
  if (await playgroundTab.count()) {
    await playgroundTab.click();
    await page.waitForTimeout(600);
  }
  const p05 = path.join(OUT, '05-playground.png');
  await page.screenshot({ path: p05 });

  // 06-pytorch-code.png
  console.log('Capturing 06-pytorch-code.png...');
  const codeTab = page.locator('button').filter({ hasText: /PyTorch Code/i }).first();
  if (await codeTab.count()) {
    await codeTab.click();
    await page.waitForTimeout(400);
  }
  const exportBtn = page.locator('button').filter({ hasText: /Export PyTorch Code/i }).first();
  if (await exportBtn.count()) {
    await exportBtn.click();
  }
  await page.waitForFunction(() => {
    const text = document.body.innerText;
    return text.includes('import torch') || text.includes('# Standalone PyTorch');
  }, { timeout: 12000 }).catch(() => null);
  await page.waitForTimeout(600);
  const p06 = path.join(OUT, '06-pytorch-code.png');
  await page.screenshot({ path: p06 });

  // 07-parameters-breakdown.png
  console.log('Capturing 07-parameters-breakdown.png...');
  const paramsTab = page.locator('button').filter({ hasText: /^Parameters/i }).first();
  if (await paramsTab.count()) {
    await paramsTab.click();
    await page.waitForTimeout(600);
  }
  const p07 = path.join(OUT, '07-parameters-breakdown.png');
  await page.screenshot({ path: p07 });

  // 08-arch4-mlp.png
  console.log('Capturing 08-arch4-mlp.png...');
  await closeDrawerIfOpen(page);
  await loadSample(page, 'Arch 4');
  const p08 = path.join(OUT, '08-arch4-mlp.png');
  await page.screenshot({ path: p08 });

  // 09-inspector-node.png
  console.log('Capturing 09-inspector-node.png...');
  await closeDrawerIfOpen(page);
  await loadSample(page, 'Arch 1');
  const firstNode = page.locator('.react-flow__node').first();
  if (await firstNode.count()) {
    await firstNode.click({ force: true });
    await page.waitForTimeout(600);
  }
  const p09 = path.join(OUT, '09-inspector-node.png');
  await page.screenshot({ path: p09 });

  // 10-context-menu.png
  console.log('Capturing 10-context-menu.png...');
  await closeDrawerIfOpen(page);
  // Click canvas pane at (400, 300) with right button
  const pane = page.locator('.react-flow__pane').first();
  if (await pane.count()) {
    await pane.click({ button: 'right', position: { x: 400, y: 300 } });
    await page.waitForSelector('[role="menu"][aria-label="Canvas context menu"]', { timeout: 6000 }).catch(() => null);
    await page.waitForTimeout(400);
  }
  const p10 = path.join(OUT, '10-context-menu.png');
  await page.screenshot({ path: p10 });

  await browser.close();

  // Delete leftover unused PNGs in docs/images/x/
  console.log('Cleaning up unused screenshots in docs/images/x/...');
  const dirFiles = await readdir(OUT);
  for (const file of dirFiles) {
    if (file.endsWith('.png') && !EXPECTED_FILES.has(file)) {
      console.log(`Removing leftover: ${file}`);
      await unlink(path.join(OUT, file));
    }
  }

  // Validate all expected files exist and > 20 KB
  console.log('Validating output files...');
  for (const file of EXPECTED_FILES) {
    const fPath = path.join(OUT, file);
    const s = await stat(fPath);
    if (s.size < 20 * 1024) {
      throw new Error(`File ${file} is unexpectedly small: ${s.size} bytes (< 20 KB)`);
    }
    console.log(`  OK: ${file} (${Math.round(s.size / 1024)} KB)`);
  }
  const spStat = await stat(SOCIAL_PREVIEW);
  console.log(`  OK: social-preview.jpg (${Math.round(spStat.size / 1024)} KB)`);

  console.log('Screenshot capture complete!');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
