export const useDarkStore = (selector?: (state: any) => any) =>
  selector
    ? selector({
        dark: false,
        stars: 0,
        version: "",
        latestVersion: "",
        discordCount: 0,
        refreshLatestVersion: () => {},
        setDark: () => {},
        refreshVersion: () => {},
        refreshStars: () => {},
        refreshDiscordCount: () => {},
      })
    : {};
