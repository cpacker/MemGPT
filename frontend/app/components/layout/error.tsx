import React from "react";
import { GITHUB_URL } from "../../constant";
import Locale from "../../locales";
import { downloadAs } from "../../utils/download";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/app/components/ui/alert-dialog";
import { cn } from "@/app/lib/utils";
import { Button, buttonVariants } from "@/app/components/ui/button";
import { Github, RefreshCcw } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/app/components/ui/card";

interface IErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  info: React.ErrorInfo | null;
}

export class ErrorBoundary extends React.Component<any, IErrorBoundaryState> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null, info: null };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Update state with error details
    this.setState({ hasError: true, error, info });
  }

  clearAndSaveData() {
    try {
      downloadAs(JSON.stringify(localStorage), "chat-llamaindex-snapshot.json");
    } finally {
      localStorage.clear();
      location.reload();
    }
  }

  render() {
    if (this.state.hasError) {
      // Render error message
      return (
        <div className="flex items-center justify-center w-screen h-screen">
          <Card className="bg-background text-sm rounded-md w-4/5 mx-auto">
            <CardHeader>
              <CardTitle>Oops, something went wrong!</CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              <pre className="overflow-auto text-xs h-[500px]">
                <code>{this.state.error?.toString()}</code>
                <code>{this.state.info?.componentStack}</code>
              </pre>
              <div className="flex justify-between">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.open(GITHUB_URL, "_blank")}
                >
                  <Github className="mr-2 h-4 w-4" />
                  <span>Report This Error</span>
                </Button>
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button variant="outline">
                      <RefreshCcw className="mr-2 h-4 w-4" />
                      <span>Clear All Data</span>
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>
                        {Locale.Settings.Danger.Clear.Confirm}
                      </AlertDialogTitle>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Cancel</AlertDialogCancel>
                      <AlertDialogAction
                        className={cn(
                          buttonVariants({ variant: "destructive" }),
                        )}
                        onClick={() => {
                          this.clearAndSaveData();
                        }}
                      >
                        Continue
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </CardContent>
          </Card>
        </div>
      );
    }
    // if no error occurred, render children
    return this.props.children;
  }
}
