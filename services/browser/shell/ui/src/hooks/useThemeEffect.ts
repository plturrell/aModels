import { useEffect } from "react";

import { useShellStore } from "../state/useShellStore";

export function useThemeEffect() {
  const theme = useShellStore((state) => state.theme);

  useEffect(() => {
    document.body.classList.remove("theme-dark", "theme-light");
    document.body.classList.add(`theme-${theme}`);
  }, [theme]);
}
