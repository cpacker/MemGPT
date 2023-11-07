import React, { PropsWithChildren } from 'react';
import { cnH1, cnLead } from '@memgpt/components/typography';

const Header = ({children}: PropsWithChildren) => (
  <div className="flex justify-between items-start">
    <div>
      <h1 className={cnH1()}>MemGPT</h1>
      <p className={cnLead()}>Towards LLMs as Operating Systems</p>
    </div>
   <div>
     {children}
   </div>
  </div>
);

export default Header;
