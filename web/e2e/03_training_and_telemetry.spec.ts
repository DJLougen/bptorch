import { test, expect } from '@playwright/test';

test.describe('Live Training, Telemetry & WebSocket Streaming E2E', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('executes single step batch and updates loss metrics', async ({ page }) => {
    // Open the bottom drawer
    const lossTabBtn = page.locator('button:has-text("Live Loss")').first();
    await lossTabBtn.click();

    // Click "Step Batch" and wait for response
    const stepBatchBtn = page.locator('button:has-text("Step Batch")').first();
    await expect(stepBatchBtn).toBeVisible();

    const responsePromise = page.waitForResponse((r) => r.url().includes('/step-batch') && r.status() === 200);
    await stepBatchBtn.click();
    await responsePromise;

    // Verify metrics dashboard displays training telemetry
    const metricsTabBtn = page.locator('button:has-text("Metrics Dashboard")').first();
    await metricsTabBtn.click();

    await expect(page.locator('text=Training Step')).toBeVisible();
    await expect(page.locator('text=Current Loss')).toBeVisible();
    await expect(page.locator('div:has-text("Learning Rate")').last()).toBeVisible();
  });

  test('adjusts live hyperparameter sliders in property inspector', async ({ page }) => {
    // Click on canvas background to deselect nodes and show model config
    await page.locator('.react-flow__pane').click({ position: { x: 50, y: 50 } });

    // Verify Property Inspector displays live hyperparameter tweakers
    const inspector = page.locator('aside').last();
    await expect(inspector.locator('text=Live Hyperparameter Tweakers')).toBeVisible();

    // Adjust learning rate slider
    const lrSlider = inspector.locator('input[type="range"]').first();
    await expect(lrSlider).toBeVisible();
    await lrSlider.fill('-2.5');

    // Adjust weight decay slider
    const wdSlider = inspector.locator('input[type="range"]').nth(1);
    await expect(wdSlider).toBeVisible();
    await wdSlider.fill('0.2');
  });

  test('runs multi-step training loop and renders live loss plotter curve', async ({ page }) => {
    // Open Live Loss Plotter
    const lossTabBtn = page.locator('button:has-text("Live Loss")').first();
    await lossTabBtn.click();

    const stepBatchBtn = page.locator('button:has-text("Step Batch")').first();
    await expect(stepBatchBtn).toBeVisible();

    // Step Batch 1
    const resp1 = page.waitForResponse((r) => r.url().includes('/step-batch') && r.status() === 200);
    await stepBatchBtn.click();
    await resp1;

    // Step Batch 2 (generates lossHistory >= 2 so plot renders)
    const resp2 = page.waitForResponse((r) => r.url().includes('/step-batch') && r.status() === 200);
    await stepBatchBtn.click();
    await resp2;

    // Verify SVG loss curve renders and stats appear
    await expect(page.locator('text=Live Loss Stats:').first()).toBeVisible({ timeout: 5000 });
    await expect(page.locator('svg').filter({ has: page.locator('path[stroke="#22c55e"]') }).first()).toBeVisible();
  });
});
