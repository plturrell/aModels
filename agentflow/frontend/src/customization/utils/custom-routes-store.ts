import { create } from 'zustand';

interface CustomRoutesState {
  customRoutes: any[];
  setCustomRoutes: (routes: any[]) => void;
}

const useCustomRoutesStore = create<CustomRoutesState>((set) => ({
  customRoutes: [],
  setCustomRoutes: (routes) => set({ customRoutes: routes }),
}));

export default useCustomRoutesStore;
