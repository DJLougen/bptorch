import { test, expect } from '@playwright/test';

test.describe('Workspace Navigation & Architecture Hierarchy E2E', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('switches templates between nanoGPT and Two-Layer MLP', async ({ page }) => {
    // Select Two-Layer MLP template
    const templateSelect = page.locator('header select').first();
    await templateSelect.selectOption({ label: 'Template: Two-Layer MLP' });

    // Verify workspace updates to Two-Layer MLP
    await expect(page.locator('header')).toContainText('Two-Layer MLP');
    await expect(page.locator('text=MLP').first()).toBeVisible();

    // Switch back to nanoGPT
    await templateSelect.selectOption({ label: 'Template: nanoGPT' });
    await expect(page.locator('header')).toContainText('nanoGPT Architecture');
    await expect(page.locator('text=nanoGPT').first()).toBeVisible();
  });

  test('drills down into composite subgraphs via Open Internals and navigates back', async ({ page }) => {
    // Find the "Open Internals" button on composite nodes (Input Embeddings or Transformer Stack)
    const openBtn = page.locator('button:has-text("Open Internals")').first();
    await expect(openBtn).toBeVisible();
    await openBtn.click();

    // Verify breadcrumb expands with sub-module name
    await expect(
      page.locator('text=Input Embeddings').or(page.locator('text=Transformer Stack')).first()
    ).toBeVisible();

    // Verify sub-graph internal nodes are rendered on canvas
    await expect(page.locator('.react-flow__node').first()).toBeVisible();

    // Navigate back using the root breadcrumb link
    const rootBreadcrumb = page.locator('text=nanoGPT').first();
    await expect(rootBreadcrumb).toBeVisible();
    await rootBreadcrumb.click();

    await expect(page.locator('header')).toContainText('nanoGPT Architecture');
  });

  test('toggles Event Graph view and Architecture view', async ({ page }) => {
    // Switch to Event Graph
    const eventGraphBtn = page.locator('button:has-text("Event Graph")');
    await expect(eventGraphBtn).toBeVisible();
    await eventGraphBtn.click();

    // Assert Event Graph nodes or breadcrumb appear
    await expect(
      page.locator('text=Event OnTrainBegin').or(page.locator('text=Training Event Graph')).or(page.locator('text=Training Sequence')).first()
    ).toBeVisible();

    // Switch back to Architecture
    const archBtn = page.locator('button:has-text("Architecture")');
    await expect(archBtn).toBeVisible();
    await archBtn.click();

    // Assert Architecture Graph nodes appear
    await expect(
      page.locator('text=Token IDs Input').or(page.locator('text=Final LayerNorm')).or(page.locator('text=Input Embeddings')).first()
    ).toBeVisible();
  });

  test('filters node palette with search query', async ({ page }) => {
    const searchInput = page.locator('input[placeholder="Search nodes..."]');
    await expect(searchInput).toBeVisible();

    await searchInput.fill('Linear');
    await expect(page.locator('aside').first().locator('text=Linear').first()).toBeVisible();

    // Clear search
    await searchInput.fill('');
    await expect(page.locator('text=FLOW CONTROL')).toBeVisible();
  });

  test('exports, reimports, and restores an authored project after reload', async ({ page }) => {
    await expect(page.locator('header')).toContainText('nanoGPT Architecture');
    const savedProjectJson = await page.evaluate(() =>
      localStorage.getItem('bptorch.project.v1')
    );
    if (!savedProjectJson) {
      throw new Error('Expected the bundled project to be persisted before import');
    }

    const importedProject = JSON.parse(savedProjectJson);
    importedProject.project.id = 'e2e_imported_project';
    importedProject.project.name = 'Reloaded Imported Project';
    await page.getByLabel('Import project JSON file').setInputFiles({
      name: 'authored-project.nbp.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(importedProject)),
    });
    await expect(page.locator('header')).toContainText('Reloaded Imported Project');

    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export project JSON' }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe('reloaded-imported-project.nbp.json');
    const downloadPath = await download.path();
    if (!downloadPath) {
      throw new Error('Expected the exported project download to have a local path');
    }

    await page.getByLabel('Project template').selectOption('linear_mlp');
    await expect(page.locator('header')).toContainText('Two-Layer MLP');
    await page.getByLabel('Import project JSON file').setInputFiles(downloadPath);
    await expect(page.locator('header')).toContainText('Reloaded Imported Project');

    await page.reload();
    await expect(page.locator('header')).toContainText('Reloaded Imported Project');
  });
});
