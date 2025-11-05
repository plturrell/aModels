import type { PropsWithChildren, ReactNode } from "react";

import styles from "./ShellLayout.module.css";

interface ShellLayoutProps {
  nav: ReactNode;
}

export function ShellLayout({ nav, children }: PropsWithChildren<ShellLayoutProps>) {
  return (
    <div className={styles.root}>
      <aside className={styles.sidebar}>{nav}</aside>
      <main className={styles.content}>{children}</main>
    </div>
  );
}
