import { create } from "zustand";

export type ShellTheme = "dark" | "light";

export type ShellModuleId = "home" | "localai" | "telemetry";

interface ShellState {
  theme: ShellTheme;
  activeModule: ShellModuleId;
  setTheme: (theme: ShellTheme) => void;
  toggleTheme: () => void;
  setActiveModule: (module: ShellModuleId) => void;
}

export const useShellStore = create<ShellState>((set) => ({
  theme: "dark",
  activeModule: "localai",
  setTheme: (theme) => set({ theme }),
  toggleTheme: () =>
    set((state) => ({
      theme: state.theme === "dark" ? "light" : "dark"
    })),
  setActiveModule: (module) => set({ activeModule: module })
}));
