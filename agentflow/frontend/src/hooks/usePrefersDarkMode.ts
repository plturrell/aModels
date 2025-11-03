import { useEffect, useState } from "react";

export function usePrefersDarkMode() {
  const [isDark, setIsDark] = useState(() =>
    typeof window !== "undefined"
      ? window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false
      : false,
  );

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) {
      return undefined;
    }
    const media = window.matchMedia("(prefers-color-scheme: dark)");

    const listener = (event: MediaQueryListEvent) => setIsDark(event.matches);
    media.addEventListener?.("change", listener);

    return () => media.removeEventListener?.("change", listener);
  }, []);

  return isDark;
}
