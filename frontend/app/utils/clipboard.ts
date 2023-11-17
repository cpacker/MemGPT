import Locale from "../locales";

type DisplayResultInput = {
  title: string;
  variant: "success" | "destructive" | "default";
};

export type DisplayResultFn = (input: DisplayResultInput) => void;

export async function copyToClipboard(
  text: string,
  displayResult: DisplayResultFn,
) {
  try {
    await navigator.clipboard.writeText(text);

    displayResult({
      title: Locale.Copy.Success,
      variant: "success",
    });
  } catch (error) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
      document.execCommand("copy");
      displayResult({
        title: Locale.Copy.Success,
        variant: "success",
      });
    } catch (error) {
      displayResult({
        title: Locale.Copy.Failed,
        variant: "destructive",
      });
    }
    document.body.removeChild(textArea);
  }
}
