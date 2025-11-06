import { ShellLayout } from "./components/ShellLayout";
import { NavPanel } from "./components/NavPanel";
import { useThemeEffect } from "./hooks/useThemeEffect";
import { useShellStore, type ShellModuleId } from "./state/useShellStore";

import { HomeModule } from "./modules/Home/HomeModule";
import { LocalAIModule } from "./modules/LocalAI/LocalAIModule";
import { DocumentsModule } from "./modules/Documents/DocumentsModule";
import { FlowsModule } from "./modules/Flows/FlowsModule";
import { TelemetryModule } from "./modules/Telemetry/TelemetryModule";

import styles from "./App.module.css";

const renderModule = (moduleId: ShellModuleId) => {
  switch (moduleId) {
    case "localai":
      return <LocalAIModule />;
    case "dms":
      return <DocumentsModule />;
    case "flows":
      return <FlowsModule />;
    case "telemetry":
      return <TelemetryModule />;
    case "home":
    default:
      return <HomeModule />;
  }
};

export default function App() {
  useThemeEffect();
  const activeModule = useShellStore((state) => state.activeModule);

  return (
    <ShellLayout nav={<NavPanel />}>
      <div className={styles.main}>{renderModule(activeModule)}</div>
    </ShellLayout>
  );
}
