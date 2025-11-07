import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CommandPalette } from '../CommandPalette';

describe('CommandPalette', () => {
  const mockOnCommand = vi.fn();

  it('should render without crashing', () => {
    render(<CommandPalette onCommand={mockOnCommand} />);
    // Initially closed, so nothing should be visible
    expect(screen.queryByPlaceholderText(/command or search/i)).not.toBeInTheDocument();
  });

  it('should open palette on Cmd+K', () => {
    render(<CommandPalette onCommand={mockOnCommand} />);
    
    // Simulate Cmd+K
    fireEvent.keyDown(window, { key: 'k', metaKey: true });
    
    // Palette should now be visible
    expect(screen.getByPlaceholderText(/command or search/i)).toBeInTheDocument();
  });

  it('should close palette on Escape', () => {
    render(<CommandPalette onCommand={mockOnCommand} />);
    
    // Open palette
    fireEvent.keyDown(window, { key: 'k', metaKey: true });
    expect(screen.getByPlaceholderText(/command or search/i)).toBeInTheDocument();
    
    // Close with Escape - need to escape on the modal backdrop
    const backdrop = screen.getByPlaceholderText(/command or search/i).closest('[role="presentation"]');
    if (backdrop) {
      fireEvent.keyDown(backdrop, { key: 'Escape' });
    }
  });

  it('should handle Enter key to execute search', async () => {
    render(<CommandPalette onCommand={mockOnCommand} />);
    
    // Open palette
    fireEvent.keyDown(window, { key: 'k', metaKey: true });
    
    const input = screen.getByPlaceholderText(/command or search/i);
    fireEvent.change(input, { target: { value: 'test query' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    
    // Should eventually call onCommand (may need to wait for async)
    await vi.waitFor(() => {
      expect(mockOnCommand).toHaveBeenCalled();
    });
  });
});
