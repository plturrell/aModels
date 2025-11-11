import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import HomePage from "../homePage";

describe("HomePage", () => {
  it("shows AgentFlow catalog link when viewing flows", () => {
    render(
      <MemoryRouter initialEntries={["/flows/"]}>
        <HomePage type="flows" />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("link", { name: /Open AgentFlow Catalog/i }),
    ).toBeInTheDocument();
  });

  it("hides AgentFlow catalog link for other sections", () => {
    const { rerender } = render(
      <MemoryRouter>
        <HomePage type="components" />
      </MemoryRouter>,
    );

    expect(
      screen.queryByRole("link", { name: /Open AgentFlow Catalog/i }),
    ).not.toBeInTheDocument();

    rerender(
      <MemoryRouter>
        <HomePage type="mcp" />
      </MemoryRouter>,
    );

    expect(
      screen.queryByRole("link", { name: /Open AgentFlow Catalog/i }),
    ).not.toBeInTheDocument();
  });
});
