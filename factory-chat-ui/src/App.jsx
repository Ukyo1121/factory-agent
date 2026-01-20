import { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Send, Plus, MessageSquare, User, Bot, Loader2, StopCircle, Zap, Wrench, AlertTriangle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Database } from 'lucide-react'; // 引入图标
import KnowledgeModal from './components/KnowledgeModal'; // 引入组件

// 后端 API 地址
const API_URL = "http://localhost:8000/chat";

function App() {
  // --- 状态管理 ---
  const [threads, setThreads] = useState([]);
  const [activeThreadId, setActiveThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isKbOpen, setIsKbOpen] = useState(false);

  // --- 打字机效果专用状态 ---
  const [streamBuffer, setStreamBuffer] = useState("");
  const [displayedContent, setDisplayedContent] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  // --- 初始化 ---
  useEffect(() => {
    if (threads.length === 0) createNewThread();
  }, []);

  // --- 自动滚动 ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, displayedContent, isLoading]);

  // --- 打字机定时器 ---
  useEffect(() => {
    if (streamBuffer.length > displayedContent.length) {
      setIsTyping(true);
      const timer = setTimeout(() => {
        setDisplayedContent(prev => streamBuffer.slice(0, prev.length + 1));
      }, 20);

      return () => clearTimeout(timer);
    } else {
      setIsTyping(false);
      if (!isLoading && streamBuffer) {
        setMessages(prev => {
          const newMsgs = [...prev];
          if (newMsgs.length > 0 && newMsgs[newMsgs.length - 1].role === 'ai') {
            newMsgs[newMsgs.length - 1].content = streamBuffer;
          }
          return newMsgs;
        });
      }
    }
  }, [streamBuffer, displayedContent, isLoading]);

  // --- 创建新会话 ---
  const createNewThread = () => {
    const newId = uuidv4();
    const newThread = { id: newId, title: "新对话", history: [] };
    setThreads(prev => [newThread, ...prev]);
    setActiveThreadId(newId);
    setMessages([]);
    resetTyper();
  };

  // --- 切换会话 ---
  const switchThread = (id) => {
    if (isLoading) return;

    if (activeThreadId) {
      setThreads(prev => prev.map(t =>
        t.id === activeThreadId ? { ...t, history: messages } : t
      ));
    }
    const targetThread = threads.find(t => t.id === id);
    if (targetThread) {
      setActiveThreadId(id);
      setMessages(targetThread.history || []);
      resetTyper();
    }
  };

  const resetTyper = () => {
    setStreamBuffer("");
    setDisplayedContent("");
    setIsTyping(false);
  };

  // --- 发送消息 ---
  const handleSend = async (manualInput = null) => {
    const textToSend = manualInput || input;
    if (!textToSend.trim() || isLoading) return;

    // 1. UI更新
    setMessages(prev => [...prev, { role: 'user', content: textToSend }]);
    setInput("");
    setIsLoading(true);
    resetTyper();

    // 2. 占位AI消息
    setMessages(prev => [...prev, { role: 'ai', content: "" }]);

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: textToSend,
          thread_id: activeThreadId
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) throw new Error("API Error");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        setStreamBuffer(prev => prev + chunk);
      }

      // 更新标题
      setThreads(prev => prev.map(t =>
        t.id === activeThreadId && t.title === "新对话"
          ? { ...t, title: textToSend }
          : t
      ));

    } catch (error) {
      if (error.name !== 'AbortError') {
        setStreamBuffer(prev => prev + "\n\n⚠️ 连接服务器失败，请检查后端。");
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-50 text-gray-800 font-sans">

      {/* 侧边栏 */}
      <div className="w-64 bg-gray-900 text-white flex flex-col flex-shrink-0">
        <div className="p-4">
          <button
            onClick={createNewThread}
            disabled={isLoading}
            className={`w-full flex items-center gap-2 bg-gray-800 p-3 rounded-md border border-gray-700 text-sm transition-colors ${isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-700'}`}
          >
            <Plus size={16} /> 新建对话
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2 custom-scrollbar">
          {threads.map(thread => (
            <button
              key={thread.id}
              onClick={() => switchThread(thread.id)}
              disabled={isLoading}
              className={`w-full text-left p-3 rounded-md mb-1 text-sm flex items-center gap-2 truncate transition-colors ${activeThreadId === thread.id ? 'bg-gray-800 text-white' : 'text-gray-400 hover:bg-gray-800'
                }`}
            >
              {/* 图标：加上 flex-shrink-0 防止被长文本挤扁 */}
              <MessageSquare size={14} className="flex-shrink-0" />

              {/* 文本：加上 truncate 实现自动省略号 */}
              <span className="truncate">{thread.title}</span>
            </button>
          ))}
        </div>

        {/* 侧边栏底部 */}
        <div className="p-4 border-t border-gray-800">
          <button
            onClick={() => setIsKbOpen(true)}
            className="w-full flex items-center gap-2 text-gray-400 hover:text-white hover:bg-gray-800 p-2 rounded-md transition-colors text-sm"
          >
            <Database size={16} />
            管理知识库
          </button>
        </div>
      </div>

      {/* 主界面 */}
      <div className="flex-1 flex flex-col relative bg-white">
        <div className="h-14 border-b flex items-center px-6 shadow-sm z-10 bg-white">
          <h1 className="font-semibold text-gray-700 flex items-center gap-2">
            <Bot className="text-blue-600" size={20} />
            工厂智能助手
            <span className="text-xs text-gray-400 font-normal px-2 py-0.5 bg-gray-100 rounded-full">Pro</span>
          </h1>
        </div>

        <div className="flex-1 overflow-y-auto p-4 pb-32 custom-scrollbar">
          <div className="max-w-3xl mx-auto space-y-6 min-h-full flex flex-col">

            {/* 🔥🔥🔥 欢迎界面 🔥🔥🔥 */}
            {messages.length === 0 && (
              <div className="flex-1 flex flex-col items-center justify-center text-center mt-10">
                <div className="w-20 h-20 bg-white rounded-2xl shadow-sm border border-gray-100 flex items-center justify-center mb-6">
                  <Bot size={40} className="text-blue-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-800 mb-3">有什么可以帮你的吗？</h2>
                <p className="text-gray-500 mb-10 max-w-md">
                  我可以帮你查询工厂设备故障、解析错误码、搜索PDF手册或提供详细的维修步骤。
                </p>

                {/* 快捷提示词卡片 */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl px-4">
                  <button
                    onClick={() => handleSend("错误码303是什么意思")}
                    className="flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left group"
                  >
                    <div className="w-10 h-10 bg-red-50 rounded-lg flex items-center justify-center group-hover:bg-blue-50 transition-colors">
                      <AlertTriangle size={20} className="text-red-500 group-hover:text-blue-600" />
                    </div>
                    <div>
                      <div className="font-semibold text-gray-700 group-hover:text-blue-700">查询错误码</div>
                      <div className="text-xs text-gray-400">错误码303是什么意思</div>
                    </div>
                  </button>

                  <button
                    onClick={() => handleSend("操作FANUC机器人时急停了怎么办")}
                    className="flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left group"
                  >
                    <div className="w-10 h-10 bg-yellow-50 rounded-lg flex items-center justify-center group-hover:bg-blue-50 transition-colors">
                      <Zap size={20} className="text-yellow-600 group-hover:text-blue-600" />
                    </div>
                    <div>
                      <div className="font-semibold text-gray-700 group-hover:text-blue-700">紧急故障</div>
                      <div className="text-xs text-gray-400">操作FANUC机器人时急停了怎么办</div>
                    </div>
                  </button>

                  <button
                    onClick={() => handleSend("自动分拣系统的操作步骤")}
                    className="flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left group"
                  >
                    <div className="w-10 h-10 bg-green-50 rounded-lg flex items-center justify-center group-hover:bg-blue-50 transition-colors">
                      <Wrench size={20} className="text-green-600 group-hover:text-blue-600" />
                    </div>
                    <div>
                      <div className="font-semibold text-gray-700 group-hover:text-blue-700">操作规程</div>
                      <div className="text-xs text-gray-400">自动分拣系统的操作步骤</div>
                    </div>
                  </button>

                  <button
                    onClick={() => handleSend("气动设备故障怎么恢复")}
                    className="flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left group"
                  >
                    <div className="w-10 h-10 bg-purple-50 rounded-lg flex items-center justify-center group-hover:bg-blue-50 transition-colors">
                      <Bot size={20} className="text-purple-600 group-hover:text-blue-600" />
                    </div>
                    <div>
                      <div className="font-semibold text-gray-700 group-hover:text-blue-700">维修指导</div>
                      <div className="text-xs text-gray-400">气动设备故障怎么恢复</div>
                    </div>
                  </button>
                </div>
              </div>
            )}

            {/* 消息列表 */}
            {messages.map((msg, idx) => {
              const isLastAiMessage = msg.role === 'ai' && idx === messages.length - 1;
              const contentToShow = isLastAiMessage && (isLoading || isTyping) ? displayedContent : msg.content;

              return (
                <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {msg.role === 'ai' && (
                    <div className="w-8 h-8 rounded-full bg-blue-50 border border-blue-100 flex items-center justify-center flex-shrink-0 mt-1">
                      <Bot size={16} className="text-blue-600" />
                    </div>
                  )}

                  <div className={`max-w-[85%] p-4 rounded-2xl text-sm leading-7 shadow-sm ${msg.role === 'user'
                    ? 'bg-blue-600 text-white rounded-br-none'
                    : 'bg-white border border-gray-100 text-gray-800 rounded-bl-none'
                    }`}>
                    {msg.role === 'ai' ? (
                      <div>
                        {(!contentToShow && isLoading) ? (
                          <div className="flex items-center gap-2 text-gray-400 py-1">
                            <Loader2 size={16} className="animate-spin" />
                            <span className="text-xs">正在思考并检索知识库...</span>
                          </div>
                        ) : (
                          <ReactMarkdown
                            components={{
                              ul: ({ node, ...props }) => <ul className="list-disc pl-4 my-2 space-y-1" {...props} />,
                              ol: ({ node, ...props }) => <ol className="list-decimal pl-4 my-2 space-y-1" {...props} />,
                              strong: ({ node, ...props }) => <span className="font-bold text-blue-700 bg-blue-50 px-1 rounded" {...props} />,
                              h1: ({ node, ...props }) => <h1 className="text-xl font-bold my-3 border-b pb-2" {...props} />,
                              h2: ({ node, ...props }) => <h2 className="text-lg font-bold my-2" {...props} />,
                              code: ({ node, inline, className, children, ...props }) => {
                                return inline ? (
                                  <code className="bg-gray-100 text-red-500 px-1 py-0.5 rounded text-xs font-mono" {...props}>{children}</code>
                                ) : (
                                  <pre className="bg-gray-800 text-gray-100 p-3 rounded-lg overflow-x-auto my-2 text-xs font-mono">
                                    <code {...props}>{children}</code>
                                  </pre>
                                )
                              },
                              // 自定义图片渲染
                              img: ({ node, ...props }) => (
                                <img
                                  {...props}
                                  className="max-w-full h-auto rounded-lg shadow-md my-4 border border-gray-200 cursor-zoom-in"
                                  onClick={() => window.open(props.src, '_blank')} // 点击在新窗口打开大图
                                  alt="操作示意图"
                                />
                              )
                            }}
                          >
                            {contentToShow}
                          </ReactMarkdown>
                        )}
                        {isLastAiMessage && (isLoading || isTyping) && (
                          <span className="inline-block w-1.5 h-4 ml-1 align-middle bg-blue-600 animate-pulse"></span>
                        )}
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                    )}
                  </div>

                  {msg.role === 'user' && (
                    <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0 mt-1">
                      <User size={16} className="text-gray-500" />
                    </div>
                  )}
                </div>
              );
            })}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* 输入框区域 */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pt-12 pb-6 px-4">
          <div className="max-w-3xl mx-auto relative group">
            <div className="bg-white border border-gray-300 rounded-xl shadow-lg flex items-end p-2 focus-within:ring-2 focus-within:ring-blue-500/20 focus-within:border-blue-400 transition-all">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend(null);
                  }
                }}
                placeholder="描述故障现象 (如: 机械臂抖动) 或输入错误码..."
                className="w-full max-h-32 bg-transparent border-none focus:ring-0 resize-none p-3 text-gray-700 placeholder-gray-400 text-sm"
                rows={1}
                disabled={isLoading}
              />

              {isLoading ? (
                <button
                  onClick={handleStop}
                  className="p-2 rounded-lg mb-1 mr-1 bg-red-50 text-red-500 hover:bg-red-100 transition-colors"
                >
                  <StopCircle size={20} />
                </button>
              ) : (
                <button
                  onClick={() => handleSend(null)}
                  disabled={!input.trim()}
                  className={`p-2 rounded-lg mb-1 mr-1 transition-all ${input.trim()
                    ? 'bg-blue-600 text-white hover:bg-blue-700 shadow-md'
                    : 'bg-gray-100 text-gray-300 cursor-not-allowed'
                    }`}
                >
                  <Send size={18} />
                </button>
              )}
            </div>
            <p className="text-center text-xs text-gray-400 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
              AI 内容由 Factory Agent 生成，仅供参考
            </p>
          </div>
        </div>
      </div>
      <KnowledgeModal isOpen={isKbOpen} onClose={() => setIsKbOpen(false)} />
    </div>
  );
}

export default App;