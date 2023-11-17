// To store AbortControllers per session
export const ChatControllerPool = {
  controllers: {} as Record<string, AbortController>,

  addController(sessionId: string, controller: AbortController) {
    this.controllers[sessionId] = controller;
  },

  stop(sessionId: string) {
    const controller = this.controllers[sessionId];
    controller?.abort();
  },

  isRunning(sessionId: string) {
    return this.controllers[sessionId] !== undefined;
  },

  remove(sessionId: string) {
    delete this.controllers[sessionId];
  },
};
