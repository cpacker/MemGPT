import { create } from 'zustand';
import { combine } from 'zustand/middleware';

const useNextMessageLoadingStore = create(combine({
    isLoading: false,
  },
  (set) => ({
    actions: {
      setLoading: (isLoading: boolean) => set(() => ({ isLoading })),
    },
  })));

export const useNextMessageLoading = () => useNextMessageLoadingStore(s => s.isLoading);

export const useNextMessageLoadingActions = () =>
  useNextMessageLoadingStore((s) => s.actions);
