import { useEffect, useState } from 'react';

interface FontStatus {
  inter: boolean;
  jetBrainsMono: boolean;
  sapIcons: boolean;
  allLoaded: boolean;
}

export const useFontObserver = () => {
  const [fontStatus, setFontStatus] = useState<FontStatus>({
    inter: false,
    jetBrainsMono: false,
    sapIcons: false,
    allLoaded: false,
  });

  useEffect(() => {
    if (!document.fonts) {
      // Fallback for older browsers
      setFontStatus({
        inter: true,
        jetBrainsMono: true,
        sapIcons: true,
        allLoaded: true,
      });
      return;
    }

    const checkFonts = async () => {
      const fonts = [
        { name: 'Inter', key: 'inter' as const },
        { name: 'JetBrains Mono', key: 'jetBrainsMono' as const },
        { name: 'SAP-icons', key: 'sapIcons' as const },
      ];

      const results = await Promise.allSettled(
        fonts.map(async (font) => {
          try {
            await document.fonts.load(`16px ${font.name}`);
            return { key: font.key, loaded: true };
          } catch {
            return { key: font.key, loaded: false };
          }
        })
      );

      const newStatus: Partial<FontStatus> = {};
      let allLoaded = true;

      results.forEach((result) => {
        if (result.status === 'fulfilled') {
          newStatus[result.value.key] = result.value.loaded;
          if (!result.value.loaded) allLoaded = false;
        } else {
          newStatus[result.value.key] = false;
          allLoaded = false;
        }
      });

      setFontStatus({
        ...newStatus,
        allLoaded,
      } as FontStatus);
    };

    checkFonts();

    // Listen for font loading events
    const handleFontLoad = () => {
      checkFonts();
    };

    document.fonts.addEventListener('loadingdone', handleFontLoad);
    document.fonts.addEventListener('loadingerror', handleFontLoad);

    return () => {
      document.fonts.removeEventListener('loadingdone', handleFontLoad);
      document.fonts.removeEventListener('loadingerror', handleFontLoad);
    };
  }, []);

  return fontStatus;
};
