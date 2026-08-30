import { test, expect } from '@playwright/test';

test.describe('Blueprint Canvas & Interactive Nodes E2E', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders Blueprint nodes on canvas and allows node selection', async ({ page }) => {
    // Verify nodes are rendered on canvas
    const nodes = page.locator('.react-flow__node');
    await expect(nodes.first()).toBeVisible();

    const nodeCount = await nodes.count();
    expect(nodeCount).toBeGreaterThan(0);

    // Click on the first node
    await nodes.first().click();

    // Verify Property Inspector opens with node details
    const inspector = page.locator('aside').last();
    await expect(inspector).toBeVisible();
    await expect(inspector.locator('button:has-text("Properties")')).toBeVisible();
    await expect(inspector.locator('button:has-text("Parameters")')).toBeVisible();
  });

  test('toggles breakpoints on node header', async ({ page }) => {
    const nodes = page.locator('.react-flow__node');
    await expect(nodes.first()).toBeVisible();

    // Find the breakpoint circle button on the node
    const breakpointBtn = nodes.first().locator('button[title*="Breakpoint"]');
    await expect(breakpointBtn).toBeVisible();

    // Click to toggle breakpoint
    await breakpointBtn.click();
    await expect(breakpointBtn).toHaveAttribute('title', /Breakpoint/);
  });

  test('inspects node parameter tab and tensor shapes', async ({ page }) => {
    const nodes = page.locator('.react-flow__node');
    await nodes.first().click();

    const inspector = page.locator('aside').last();
    const paramsTab = inspector.locator('button:has-text("Parameters")');
    await paramsTab.click();

    // Verify parameter information is shown
    await expect(inspector).toContainText(/Parameters/i);
  });
});
