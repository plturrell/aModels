import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

const mockNavigate = jest.fn();

jest.mock("react-router-dom", () => ({
  ...jest.requireActual("react-router-dom"),
  useNavigate: () => mockNavigate,
}));

import AgentFlowPage from "../agentFlowPage";
import {
  fetchAgentFlowCatalog,
  importAgentFlow,
} from "../../../../controllers/API/agentflowAPI";

jest.mock("../../../../controllers/API/agentflowAPI", () => ({
  fetchAgentFlowCatalog: jest.fn(),
  importAgentFlow: jest.fn(),
}));

const mockFetchCatalog = fetchAgentFlowCatalog as jest.MockedFunction<
  typeof fetchAgentFlowCatalog
>;
const mockImportFlow = importAgentFlow as jest.MockedFunction<typeof importAgentFlow>;

describe("AgentFlowPage", () => {
  afterEach(() => {
    jest.clearAllMocks();
    mockNavigate.mockReset();
  });

  it("renders catalog entries", async () => {
    mockFetchCatalog.mockResolvedValue({
      total: 1,
      page: 1,
      page_size: 10,
      items: [
        {
          id: "flow_one",
          name: "Flow One",
          category: "processes",
          description: "Sample description",
          relative_path: "processes/flow_one.json",
        },
      ],
    });

    await act(async () => {
      render(<AgentFlowPage />);
    });

    expect(
      screen.getByRole("heading", { name: /AgentFlow Catalog/i }),
    ).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("Flow One")).toBeInTheDocument();
    });
  });

  it("imports a flow when the action button is clicked", async () => {
    mockFetchCatalog.mockResolvedValue({
      total: 1,
      page: 1,
      page_size: 10,
      items: [
        {
          id: "flow_one",
          name: "Flow One",
          category: "processes",
          description: "Sample description",
          relative_path: "processes/flow_one.json",
        },
      ],
    });
    mockImportFlow.mockResolvedValue({
      imported: true,
      flow: { id: "remote-id", name: "Flow One" },
    });

    await act(async () => {
      render(<AgentFlowPage />);
    });

    const importButton = await screen.findByRole("button", { name: /import/i });
    fireEvent.click(importButton);

    await waitFor(() => {
      expect(mockImportFlow).toHaveBeenCalledWith("flow_one");
      expect(mockNavigate).toHaveBeenCalledWith("/flow/remote-id/");
    });
  });
});
