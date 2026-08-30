import { test, expect } from '@playwright/test';

test.describe('Architecture Tester & Automated Evaluation E2E', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('opens Architecture Tester tab, runs full 6-pillar test suite, and displays pass badges', async ({ page }) => {
    // 1. Open bottom drawer on Architecture Tester tab
    const testerTabBtn = page.locator('button:has-text("Architecture Tester")').or(page.locator('button:has-text("Tester")')).first();
    await expect(testerTabBtn).toBeVisible();
    await testerTabBtn.click();

    // 2. Click "Run Test Suite (6 Tests)" and wait for /test/run API response
    const runSuiteBtn = page.locator('button:has-text("Run Test Suite")').first();
    await expect(runSuiteBtn).toBeVisible();

    const responsePromise = page.waitForResponse((r) => r.url().includes('/test/run') && r.status() === 200, { timeout: 30000 });
    await runSuiteBtn.click();
    await responsePromise;

    // 3. Verify Passed summary
    await expect(page.locator('text=/\\d+\\/\\d+ Passed/').first()).toBeVisible({ timeout: 10000 });

    // 4. Verify individual test case cards
    await expect(page.locator('text=Dynamic Shape & Forward Pass Sanity')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('text=Autograd & Gradient Flow Health')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('text=Optimization & Single-Batch Convergence')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('text=Stateful Checkpoint Save & Restore')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('text=Standalone Cooking & Subprocess Dry-Run')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('text=Static Schema & Numerical Stability')).toBeVisible({ timeout: 5000 });

    // 5. Verify at least 6 PASSED chips are displayed
    const count = await page.locator('span:has-text("PASSED")').count();
    expect(count).toBeGreaterThanOrEqual(6);
  });
});
