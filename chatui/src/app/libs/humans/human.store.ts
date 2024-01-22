import { create } from 'zustand';
import { Human } from './human';

interface HumanStoreState {
    humans: Human[];
    setHumans: (humans: Human[]) => void;
}

export const useHumanStore = create<HumanStoreState>((set) => ({
    humans: [],
    setHumans: (humans) => set({ humans }),
}));