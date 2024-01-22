import { create } from 'zustand';
import { Model } from './model';

interface ModelStoreState {
    models: Model[];
    setModels: (models: Model[]) => void;
}

export const useModelStore = create<ModelStoreState>((set) => ({
    models: [],
    setModels: (models) => set({ models }),
}));