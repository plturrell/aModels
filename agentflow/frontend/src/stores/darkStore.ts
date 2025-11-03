import { create } from 'zustand';

const useDarkStore = create((set) => ({
  dark: false,
  setDark: (dark) => set({ dark }),
}));

export default useDarkStore;
