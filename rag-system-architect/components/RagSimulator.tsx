import React, { useState } from 'react';
import { Send, Loader2, Database, Bot, FileText } from 'lucide-react';
import clsx from 'clsx';

const RagSimulator: React.FC = () => {
  const [ticketInput, setTicketInput] = useState("User is asking about how to reset their SSO password when the email link expires.");
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<string | null>(null);

  const steps = [
    { delay: 800, msg: "Webhook received. Triggering background task..." },
    { delay: 1600, msg: "Bedrock: Generating embeddings for ticket..." },
    { delay: 2400, msg: "OpenSearch: Querying vector index 'zendesk-knowledge-base'..." },
    { delay: 3000, msg: "Retrieved 3 relevant articles: 'SSO Troubleshooting', 'Password Policy', 'Login Errors'." },
    { delay: 3800, msg: "Bedrock: Sending context + query to Claude 3..." },
    { delay: 5500, msg: "Zendesk API: Posting internal note..." },
  ];

  const runSimulation = () => {
    if (!ticketInput.trim()) return;
    setIsProcessing(true);
    setLogs([]);
    setResult(null);

    let totalDelay = 0;
    steps.forEach((step, index) => {
      totalDelay += step.delay - (index > 0 ? steps[index - 1].delay : 0);
      setTimeout(() => {
        setLogs(prev => [...prev, step.msg]);
        if (index === steps.length - 1) {
          setIsProcessing(false);
          setResult(`Hi there,

Based on our internal documentation, here are the steps to resolve an expired SSO password link:

1. Go to the SSO portal settings page.
2. Click "Forgot Password" again to trigger a fresh link.
3. Ensure you click the link within 15 minutes.

If the issue persists, please check your spam folder or contact the IT administrator directly.

Best regards,
AI Assistant`);
        }
      }, step.delay);
    });
  };

  return (
    <div className="p-8 h-full bg-gray-50 overflow-y-auto">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Pipeline Simulator</h2>
          <p className="text-gray-600">Test the RAG logic without deploying infrastructure.</p>
        </div>

        <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 text-sm text-indigo-800">
          <strong>💡 Demo Mode:</strong>{' '}
          Set <code className="bg-indigo-100 px-1 rounded">DEMO_MODE=true</code> in your{' '}
          <code className="bg-indigo-100 px-1 rounded">.env</code> to run the full backend pipeline with
          fake data — no AWS, OpenSearch, or Zendesk credentials needed.
          <br />
          <code className="bg-indigo-100 px-1 rounded mt-1 inline-block">
            curl -X POST http://localhost:8000/webhooks/antigravity -H "Content-Type: application/json" -d
            '{"{"}\"ticket_id\":1,\"subject\":\"Test\",\"description\":\"Hello\"{"}"}'
          </code>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Section */}
          <div className="space-y-4">
            <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
              <label className="block text-sm font-medium text-gray-700 mb-2">Simulated Ticket Description</label>
              <textarea
                className="w-full h-32 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none text-sm"
                value={ticketInput}
                onChange={(e) => setTicketInput(e.target.value)}
                placeholder="Enter a customer query..."
              />
              <div className="mt-4 flex justify-end">
                <button
                  onClick={runSimulation}
                  disabled={isProcessing || !ticketInput}
                  className={clsx(
                    "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors",
                    isProcessing ? "bg-gray-100 text-gray-400" : "bg-indigo-600 text-white hover:bg-indigo-700"
                  )}
                >
                  {isProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                  {isProcessing ? 'Processing...' : 'Simulate Webhook'}
                </button>
              </div>
            </div>

            {/* System Logs */}
            <div className="bg-gray-900 rounded-xl shadow-sm overflow-hidden border border-gray-800">
              <div className="px-4 py-2 bg-gray-800 border-b border-gray-700 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-red-500"></div>
                <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                <div className="w-2 h-2 rounded-full bg-green-500"></div>
                <span className="ml-2 text-xs text-gray-400 font-mono">system_logs</span>
              </div>
              <div className="p-4 h-48 overflow-y-auto font-mono text-xs space-y-2">
                {logs.length === 0 && <span className="text-gray-600 italic">Waiting for event...</span>}
                {logs.map((log, i) => (
                  <div key={i} className="text-green-400">
                    <span className="text-gray-500 mr-2">[{new Date().toLocaleTimeString()}]</span>
                    {log}
                  </div>
                ))}
                {isProcessing && (
                  <div className="animate-pulse text-gray-500">_</div>
                )}
              </div>
            </div>
          </div>

          {/* Output Section */}
          <div className="space-y-4">
            <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm h-full flex flex-col">
              <h3 className="text-sm font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Bot className="w-4 h-4 text-indigo-600" />
                Generated Draft Response
              </h3>

              <div className="flex-1 bg-gray-50 rounded-lg border border-gray-100 p-4">
                {!result ? (
                  <div className="h-full flex flex-col items-center justify-center text-gray-400 text-sm">
                    <FileText className="w-8 h-8 mb-2 opacity-20" />
                    <span>Response will appear here</span>
                  </div>
                ) : (
                  <div className="prose prose-sm prose-indigo animate-in fade-in duration-500">
                    <p className="whitespace-pre-wrap text-gray-700">{result}</p>
                  </div>
                )}
              </div>

              {result && (
                <div className="mt-4 pt-4 border-t border-gray-100 flex items-center justify-between text-xs text-gray-500">
                  <div className="flex items-center gap-1">
                    <Database className="w-3 h-3" />
                    <span>Context Sources: 3</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Bot className="w-3 h-3" />
                    <span>Model: Claude 3 Sonnet</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RagSimulator;
