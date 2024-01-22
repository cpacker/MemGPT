import { create } from 'zustand';
import { Persona } from './persona';

interface PersonaStoreState {
    personas: Persona[];
    setPersonas: (personas: Persona[]) => void;
}

export const usePersonaStore = create<PersonaStoreState>((set) => ({
    personas: [],
    setPersonas: (personas) => set({ personas }),
}));