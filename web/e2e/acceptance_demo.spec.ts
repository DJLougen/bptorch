import { test, expect } from '@playwright/test';

test.describe('Neural Blueprint Studio E2E Acceptance Flow', () => {
  test('launches application and displays nanoGPT blueprint workspace', async ({ page }) => {
    // 1. Visit local application
    await page.goto('/');

    // 2. Verify TopBar brand and titles
    await expect(page.locator('header')).toContainText('NEURAL BLUEPRINT');
    await expect(page.locator('header')).toContainText('nanoGPT Architecture');

    // 3. Verify Breadcrumb
    await expect(page.locator('text=nanoGPT').first()).toBeVisible();

    // 4. Verify Node Palette categories
    await expect(page.locator('aside').first().locator('text=FLOW CONTROL')).toBeVisible();
    await expect(page.locator('aside').first().locator('text=EVENTS')).toBeVisible();
    await expect(page.locator('aside').first().locator('text=DATA PIPELINES')).toBeVisible();
    await expect(page.locator('aside').first().locator('text=OPTIMIZATION')).toBeVisible();
    await expect(page.locator('aside').first().getByText('Inputs', { exact: true })).toBeVisible();
    await expect(page.locator('aside').first().getByText('Layers', { exact: true })).toBeVisible();
    await expect(page.locator('aside').first().getByText('Attention', { exact: true })).toBeVisible();

    // 5. Verify Model Configuration in Property Inspector
    await expect(page.locator('text=Model Configuration')).toBeVisible();
    await expect(page.locator('text=block size')).toBeVisible();
    await expect(page.locator('text=n embd')).toBeVisible();

    // 6. Verify Validation Status badge
    await expect(page.locator('text=Graph Valid')).toBeVisible();
    await expect(page.getByText('Bundled nanoGPT baseline parity: Verified')).toBeVisible();
  });
});
