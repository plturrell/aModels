import { create } from 'zustand';

interface CustomRoutesStorePagesState {
  customPages: any[];
  setCustomPages: (pages: any[]) => void;
}

const useCustomRoutesStorePages = create<CustomRoutesStorePagesState>((set) => ({
  customPages: [],
  setCustomPages: (pages) => set({ customPages: pages }),
}));

export default useCustomRoutesStorePages;
