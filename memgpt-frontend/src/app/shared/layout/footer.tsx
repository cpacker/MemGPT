import React from 'react';
import { cnH1, cnH4, cnLead, cnMuted } from '@memgpt/components/typography';

function Footer() {
  const year = new Date().getFullYear();
  return <div className="p-8 border-t flex justify-between items-end">
      <div>
        <p className={cnH4()}>MemGPT</p>
        <p className={cnMuted()}>Towards LLMs as Operating Systems</p>
      </div>
      <p className={cnMuted()}>&copy; {year} MemGPT</p>
    </div>;
}

export default Footer;
