import { render, screen } from '@testing-library/react';
import { HomeModule } from './HomeModule';

describe('HomeModule', () => {
  it('renders the HomeModule component', () => {
    render(<HomeModule />);
    expect(screen.getByText(/Welcome/i)).toBeInTheDocument();
  });
});
