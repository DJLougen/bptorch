import { test, expect } from '@playwright/test';

test.describe('Bottom Drawer Tabs & Diagnostics E2E', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('switches through all drawer tabs and verifies diagnostic and parity panels', async ({ page }) => {
    // 1. Diagnostics Tab
    const diagBtn = page.locator('button:has-text("Diagnostics")').first();
    await diagBtn.click();
    await expect(page.locator('text=No validation errors')).toBeVisible();

    // 2. Parameters Tab
    const paramsBtn = page.locator('button:has-text("Parameters")').first();
    await paramsBtn.click();
    await expect(page.locator('text=Unique Parameters:')).toBeVisible();
    await expect(page.locator('text=Trainable:')).toBeVisible();
    await expect(page.locator('text=Shared References:')).toBeVisible();

    // 3. Reference Parity Tab
    const parityBtn = page.locator('button:has-text("Reference Parity")');
    await parityBtn.click();
    await expect(
      page.getByText('Bundled nanoGPT baseline passed the pinned reference parity suite')
    ).toBeVisible();
    await expect(
      page.getByText(
        'This evidence applies to the bundled nanoGPT baseline, not arbitrary edited or imported projects.'
      )
    ).toBeVisible();
    await expect(page.getByText('✓ Forward logits and loss tolerances')).toBeVisible();

    // 4. Logs Tab
    const logsBtn = page.locator('button:has-text("Logs")').first();
    await logsBtn.click();
    await expect(page.locator('text=No execution logs captured yet').or(page.locator('span:has-text("[INFO]")').first())).toBeVisible();

    // 5. Close and Re-open Drawer
    const closeBtn = page.locator('button[title="Stop Run"]').locator('..').locator('button').last();
    if (await closeBtn.isVisible()) {
      await closeBtn.click();
    }
  });
});
