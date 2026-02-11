import React, { useState } from 'react';
import { File, Folder, ChevronRight, ChevronDown, Copy, Check } from 'lucide-react';
import { pythonFileSystem } from '../data/backendCode';
import { FileSystem, FileNode, FolderNode } from '../types';
import clsx from 'clsx';

const CodeBrowser: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<FileNode | null>(null);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['app', 'app/services']));
  const [copied, setCopied] = useState(false);

  // Initialize with main.py if not selected
  React.useEffect(() => {
    if (!selectedFile) {
        // Simple search for main.py
        const main = pythonFileSystem.find(n => n.name === 'main.py') as FileNode;
        if(main) setSelectedFile(main);
    }
  }, [selectedFile]);

  const toggleFolder = (path: string) => {
    const next = new Set(expandedFolders);
    if (next.has(path)) {
      next.delete(path);
    } else {
      next.add(path);
    }
    setExpandedFolders(next);
  };

  const handleCopy = () => {
    if (selectedFile) {
      navigator.clipboard.writeText(selectedFile.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const renderTree = (nodes: FileSystem, pathPrefix = '') => {
    return nodes.map((node) => {
      const currentPath = pathPrefix ? `${pathPrefix}/${node.name}` : node.name;
      
      if ('files' in node) { // It's a folder
        const isExpanded = expandedFolders.has(currentPath);
        return (
          <div key={currentPath}>
            <div 
              className="flex items-center px-2 py-1 cursor-pointer hover:bg-gray-100 text-sm text-gray-700 select-none"
              onClick={() => toggleFolder(currentPath)}
            >
              {isExpanded ? <ChevronDown className="w-4 h-4 mr-1 text-gray-400" /> : <ChevronRight className="w-4 h-4 mr-1 text-gray-400" />}
              <Folder className="w-4 h-4 mr-2 text-blue-500 fill-blue-500" />
              <span>{node.name}</span>
            </div>
            {isExpanded && (
              <div className="pl-4 border-l border-gray-200 ml-3">
                {renderTree(node.files, currentPath)}
              </div>
            )}
          </div>
        );
      } else { // It's a file
        const isSelected = selectedFile?.name === node.name && selectedFile?.content === node.content;
        return (
          <div 
            key={currentPath}
            className={clsx(
              "flex items-center px-2 py-1 cursor-pointer text-sm",
              isSelected ? "bg-indigo-50 text-indigo-700 font-medium" : "hover:bg-gray-100 text-gray-600"
            )}
            onClick={() => setSelectedFile(node)}
          >
             <span className="w-4 mr-1"></span> {/* Indent spacer for icon alignment */}
            <File className="w-4 h-4 mr-2 text-gray-400" />
            <span>{node.name}</span>
          </div>
        );
      }
    });
  };

  return (
    <div className="flex h-full bg-white border-t border-gray-200">
      {/* Sidebar */}
      <div className="w-64 flex-shrink-0 border-r border-gray-200 bg-gray-50 flex flex-col">
        <div className="p-3 border-b border-gray-200 font-semibold text-gray-700 text-sm">
          Project Files
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {renderTree(pythonFileSystem)}
        </div>
      </div>

      {/* Editor Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#1e1e1e]">
        {/* File Tab */}
        <div className="bg-[#2d2d2d] flex items-center justify-between px-4 py-2 border-b border-black">
          <span className="text-gray-300 text-sm font-mono">{selectedFile?.name}</span>
          <button 
            onClick={handleCopy}
            className="text-gray-400 hover:text-white transition-colors flex items-center gap-1 text-xs uppercase tracking-wider"
          >
            {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>

        {/* Code Content */}
        <div className="flex-1 overflow-auto p-4">
          <pre className="font-mono text-sm text-gray-300 leading-relaxed whitespace-pre">
            {selectedFile?.content || "// Select a file to view content"}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default CodeBrowser;