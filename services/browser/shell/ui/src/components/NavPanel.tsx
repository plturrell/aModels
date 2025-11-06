import { useMemo } from "react";

import { useShellStore, type ShellModuleId } from "../state/useShellStore";

import styles from "./NavPanel.module.css";

const NAV_ITEMS: Array<{ id: ShellModuleId; label: string; description: string }> = [
  {
    id: "localai",
    label: "LocalAI",
    description: "Chat on top of vendored models"
  },
  {
    id: "telemetry",
    label: "Telemetry",
    description: "Latency and usage instrumentation"
  },
  {
    id: "home",
    label: "Home",
    description: "Overview and quick links"
  }
];

export function NavPanel() {
  const activeModule = useShellStore((state) => state.activeModule);
  const setActiveModule = useShellStore((state) => state.setActiveModule);
  const theme = useShellStore((state) => state.theme);
  const toggleTheme = useShellStore((state) => state.toggleTheme);

  const orderedItems = useMemo(() => {
    const primary = NAV_ITEMS.filter((item) => item.id !== "home");
    const home = NAV_ITEMS.find((item) => item.id === "home");
    return home ? [...primary, home] : primary;
  }, []);

  return (
    <div className={styles.container}>
      <header className={styles.branding}>
        <span className={styles.glow} aria-hidden="true" />
        <div>
          <strong>aModels Shell</strong>
          <p>Chromium host for SGMI &amp; LocalAI workflows</p>
        </div>
      </header>

      <nav className={styles.nav} aria-label="Primary">
        {orderedItems.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`${styles.navItem} ${activeModule === item.id ? styles.active : ""}`}
            onClick={() => setActiveModule(item.id)}
          >
            <span>{item.label}</span>
            <small>{item.description}</small>
          </button>
        ))}
      </nav>

      <div className={styles.controls}>
        <button type="button" className={styles.themeToggle} onClick={toggleTheme}>
          Theme: {theme === "dark" ? "Dark" : "Light"}
        </button>
      </div>
    </div>
  );
}
