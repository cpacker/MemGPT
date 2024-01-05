import {
  createContext,
  Dispatch,
  PropsWithChildren,
  SetStateAction,
  useCallback,
  useContext, useEffect,
  useMemo,
  useState,
} from 'react';

const ThemeContext = createContext<{
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  setTheme: Dispatch<SetStateAction<'light' | 'dark'>>;
}>({
  setTheme(value: ((prevState: ("light" | "dark")) => ("light" | "dark")) | "light" | "dark"): void {},
  toggleTheme() {},
  theme: localStorage.getItem('theme') === 'dark' ? 'dark' : 'light'
});

export function ThemeProvider({ children }: PropsWithChildren) {
  const [theme, setTheme] = useState<'light' | 'dark'>(localStorage.getItem('theme') === 'dark' ? 'dark' : 'light');
  const toggleTheme = useCallback(() => setTheme(prev => prev === 'light' ? 'dark' : 'light'), [setTheme])
  const contextValue = useMemo(() => ({
    theme,
    setTheme,
    toggleTheme,
  }), [theme, setTheme, toggleTheme]);

  useEffect(() => {
    if (theme === 'light') {
      document.documentElement.classList.remove('dark');
      document.documentElement.classList.add('light');
    } else {
      document.documentElement.classList.remove('light');
      document.documentElement.classList.add('dark');
    }

    localStorage.setItem('theme', theme)

  }, [theme]);

  return (
    <ThemeContext.Provider value={contextValue}>{children}</ThemeContext.Provider>);
}

export const useTheme = () => useContext(ThemeContext);
