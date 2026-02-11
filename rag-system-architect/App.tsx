import React, { useState } from 'react';
import { Layers, PlayCircle, Github } from 'lucide-react';
import ArchitectureView from './components/ArchitectureView';
import RagSimulator from './components/RagSimulator';
import clsx from 'clsx';

enum Tab {
  ARCHITECTURE = 'ARCHITECTURE',
  SIMULATION = 'SIMULATION',
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>(Tab.ARCHITECTURE);

  const renderContent = () => {
    switch (activeTab) {
      case Tab.ARCHITECTURE:
        return <ArchitectureView />;
      case Tab.SIMULATION:
        return <RagSimulator />;
      default:
        return <ArchitectureView />;
    }
  };

  const NavItem = ({ tab, icon: Icon, label }: { tab: Tab, icon: any, label: string }) => (
    <button
      onClick={() => setActiveTab(tab)}
      className={clsx(
        "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors",
        activeTab === tab
          ? "bg-indigo-600 text-white shadow-sm"
          : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
      )}
    >
      <Icon className="w-4 h-4" />
      {label}
    </button>
  );

  return (
    <div className="h-screen w-screen flex flex-col bg-white overflow-hidden">
      {/* Header */}
      <header className="h-16 border-b border-gray-200 flex items-center justify-between px-6 bg-white z-10 shrink-0">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600 p-2 rounded-lg">
            <Layers className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900">RAG Architect</h1>
            <p className="text-xs text-gray-500">Zendesk + Bedrock Automation System</p>
          </div>
        </div>

        <nav className="flex items-center gap-2">
          <NavItem tab={Tab.ARCHITECTURE} icon={Layers} label="Architecture" />
          <NavItem tab={Tab.SIMULATION} icon={PlayCircle} label="Simulator" />
        </nav>

        <div className="flex items-center gap-4">
          <a href="#" className="text-gray-400 hover:text-gray-600">
            <Github className="w-5 h-5" />
          </a>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden relative">
        {renderContent()}
      </main>
    </div>
  );
}

export default App;
