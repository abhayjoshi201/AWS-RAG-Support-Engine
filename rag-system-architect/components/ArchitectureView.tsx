import React from 'react';
import { ArrowRight, Database, Server, FileText, MessageSquare, Bot, Globe } from 'lucide-react';

const StepCard = ({ icon: Icon, title, description }: { icon: any, title: string, description: string }) => (
  <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200 flex flex-col items-center text-center w-48 z-10">
    <div className="p-3 bg-indigo-50 rounded-full mb-3">
      <Icon className="w-6 h-6 text-indigo-600" />
    </div>
    <h3 className="font-semibold text-gray-900 text-sm mb-1">{title}</h3>
    <p className="text-xs text-gray-500">{description}</p>
  </div>
);

const Connector = () => (
  <div className="hidden md:flex items-center justify-center w-12 text-gray-400">
    <ArrowRight className="w-6 h-6" />
  </div>
);

const ArchitectureView: React.FC = () => {
  return (
    <div className="p-8 bg-gray-50 h-full overflow-y-auto">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">System Architecture</h2>
        <p className="text-gray-600 mb-12">
          High-level data flow for the Zendesk RAG automation system using AWS Bedrock and OpenSearch.
        </p>

        <div className="relative">
          {/* Background decoration */}
          <div className="absolute top-1/2 left-0 w-full h-1 bg-gray-200 -translate-y-1/2 hidden md:block z-0"></div>

          <div className="flex flex-col md:flex-row gap-8 justify-between relative z-10">
            <StepCard 
              icon={MessageSquare} 
              title="Zendesk" 
              description="New Ticket Event triggers webhook"
            />
            
            <Connector />

            <StepCard 
              icon={Server} 
              title="FastAPI" 
              description="Receives webhook, handles logic async"
            />
            
            <Connector />

            <div className="flex flex-col gap-4">
              <StepCard 
                icon={Bot} 
                title="Bedrock (Titan)" 
                description="Generates embeddings for ticket text"
              />
              <div className="h-8 w-1 bg-gray-200 mx-auto hidden md:block"></div>
               <StepCard 
                icon={Database} 
                title="OpenSearch" 
                description="Vector similarity search for context"
              />
               <div className="h-8 w-1 bg-gray-200 mx-auto hidden md:block"></div>
               <StepCard 
                icon={Bot} 
                title="Bedrock (Claude)" 
                description="Generates response using context"
              />
            </div>
            
             <Connector />

             <StepCard 
              icon={Globe} 
              title="Zendesk API" 
              description="Posts draft as internal note"
            />
          </div>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
            <h3 className="text-lg font-semibold mb-4 text-gray-900">Key Components</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-3">
                <div className="mt-1"><Server className="w-4 h-4 text-indigo-500"/></div>
                <div>
                  <span className="font-medium text-gray-900">FastAPI Backend</span>
                  <p className="text-sm text-gray-500">Handles webhooks asynchronously to prevent Zendesk timeouts.</p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="mt-1"><Database className="w-4 h-4 text-indigo-500"/></div>
                <div>
                  <span className="font-medium text-gray-900">AWS OpenSearch</span>
                  <p className="text-sm text-gray-500">Stores embeddings of help articles using k-NN plugin for efficient retrieval.</p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="mt-1"><Bot className="w-4 h-4 text-indigo-500"/></div>
                <div>
                  <span className="font-medium text-gray-900">AWS Bedrock</span>
                  <p className="text-sm text-gray-500">Managed access to Titan (embeddings) and Claude 3 (generation).</p>
                </div>
              </li>
            </ul>
          </div>

          <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
             <h3 className="text-lg font-semibold mb-4 text-gray-900">Production Considerations</h3>
             <ul className="space-y-3">
               <li className="text-sm text-gray-600 bg-green-50 p-2 rounded border border-green-100">
                 <strong>Retry Logic:</strong> Implemented exponential backoff for AWS API calls using `tenacity`.
               </li>
               <li className="text-sm text-gray-600 bg-blue-50 p-2 rounded border border-blue-100">
                 <strong>Security:</strong> Secrets managed via environment variables. Webhook signature verification placeholder included.
               </li>
               <li className="text-sm text-gray-600 bg-purple-50 p-2 rounded border border-purple-100">
                 <strong>Observability:</strong> Structured logging setup for tracking RAG pipeline execution flow.
               </li>
             </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ArchitectureView;
