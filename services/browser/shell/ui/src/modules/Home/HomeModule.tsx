import { Panel } from "../../components/Panel";

import styles from "./HomeModule.module.css";

const quickLinks = [
  {
    label: "SGMI Control-M overview",
    description: "Inspect the latest waits, triggers, and dummy jobs",
    targetModule: "localai"
  },
  {
    label: "Document library",
    description: "Review the latest uploads and relationships",
    targetModule: "dms"
  },
  {
    label: "Flow orchestrator",
    description: "Sync and run AgentFlow / LangFlow pipelines",
    targetModule: "flows"
  },
  {
    label: "Telemetry dashboard",
    description: "Track response latency and token usage",
    targetModule: "telemetry"
  }
];

export function HomeModule() {
  return (
    <div className={styles.home}>
      <Panel title="Welcome" subtitle="Your aModels Chromium workspace">
        <p>
          Use the navigation to explore SGMI Control-M data, inspect LocalAI models, and monitor
          telemetry captured from live interactions. Everything you see is sourced directly from the
          repository—no mock data.
        </p>
      </Panel>

      <Panel title="Quick Links" subtitle="Jump into the modules">
        <ul className={styles.links}>
          {quickLinks.map((link) => (
            <li key={link.label}>
              <span>{link.label}</span>
              <small>{link.description}</small>
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  );
}
