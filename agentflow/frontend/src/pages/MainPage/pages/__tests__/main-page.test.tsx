import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import CollectionPage from "../main-page";

describe("CollectionPage navigation", () => {
  it("renders AgentFlow link", () => {
    render(
      <MemoryRouter initialEntries={["/flows/"]}>
        <CollectionPage />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("link", { name: /AgentFlow Catalog/i }),
    ).toBeInTheDocument();
  });
});
