import type { PropsWithChildren, ReactNode } from "react";

import styles from "./Panel.module.css";

interface PanelProps {
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  dense?: boolean;
}

export function Panel({ title, subtitle, actions, dense, children }: PropsWithChildren<PanelProps>) {
  return (
    <section className={`${styles.panel} ${dense ? styles.dense : ""}`}>
      <header className={styles.header}>
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {actions ? <div className={styles.actions}>{actions}</div> : null}
      </header>
      <div className={styles.body}>{children}</div>
    </section>
  );
}
