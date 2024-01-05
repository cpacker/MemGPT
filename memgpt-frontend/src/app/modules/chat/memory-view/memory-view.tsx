import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@memgpt/components/dialog';
import { MemoryForm } from './memory-form';

const MemoryView = ({ open, onOpenChange }: { open: boolean, onOpenChange: (open: boolean) => void }) => {
  return <Dialog open={open} onOpenChange={onOpenChange}>
    <DialogContent className="sm:max-w-2xl">
      <DialogHeader>
        <DialogTitle>Edit Memory</DialogTitle>
        <DialogDescription>
          This is your agents current memory. Make changes and click save to edit it.
        </DialogDescription>
      </DialogHeader>
      <MemoryForm className="max-h-[80vh] overflow-auto px-1 py-4" />
    </DialogContent>
  </Dialog>;
};

export default MemoryView;
