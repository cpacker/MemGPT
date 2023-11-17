import { Loader2 } from "lucide-react";

export function Loading() {
  return <Loader2 className="h-4 w-4 animate-spin" />;
}

export function LoadingPage() {
  return (
    <div className="w-full h-screen max-h-full flex items-center justify-center text-sm text-muted-foreground">
      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
      Loading...
    </div>
  );
}
