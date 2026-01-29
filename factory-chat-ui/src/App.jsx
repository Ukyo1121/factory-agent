import { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  Send, Plus, MessageSquare, User, Bot, Loader2, StopCircle,
  Zap, Wrench, AlertTriangle, Database, Mic, ClipboardList, BarChart2,
  Bug, Terminal, GraduationCap, Archive
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import KnowledgeModal from './components/KnowledgeModal';
import UnansweredModal from './components/UnansweredModal';
import LifecycleDashboard from './components/LifecycleDashboard';

const API_URL = "http://localhost:8000/chat";
const VOICE_API_URL = "http://localhost:8000/voice-to-text";

function App() {
  // --- 状态管理 ---
  const [threads, setThreads] = useState([]);
  const [activeThreadId, setActiveThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // 模块弹窗状态
  const [isKbOpen, setIsKbOpen] = useState(false); // 采集助手 - 知识库
  const [isUnansweredOpen, setIsUnansweredOpen] = useState(false); // 采集助手 - 待解答
  const [isDashboardOpen, setIsDashboardOpen] = useState(false); // 生产监测 - 看板

  // 调试助手模式状态
  const [isDebugMode, setIsDebugMode] = useState(false);

  // --- 打字机效果专用状态 ---
  const [streamBuffer, setStreamBuffer] = useState("");
  const [displayedContent, setDisplayedContent] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  // 语音相关状态
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessingVoice, setIsProcessingVoice] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const [unansweredCount, setUnansweredCount] = useState(0);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  // --- 初始化 ---
  useEffect(() => {
    if (threads.length === 0) createNewThread();
    fetchUnansweredCount();
  }, []);

  useEffect(() => {
    if (!isUnansweredOpen) fetchUnansweredCount();
  }, [isUnansweredOpen]);

  const fetchUnansweredCount = async () => {
    try {
      const res = await fetch("http://localhost:8000/admin/unanswered_questions");
      const data = await res.json();
      setUnansweredCount(data.count || 0);
    } catch (e) {
      console.error("获取待解答数量失败", e);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, displayedContent, isLoading]);

  // --- 打字机逻辑 ---
  useEffect(() => {
    if (streamBuffer.length > displayedContent.length) {
      setIsTyping(true);
      const timer = setTimeout(() => {
        setDisplayedContent(prev => streamBuffer.slice(0, prev.length + 1));
      }, 10); // 略微加快打字速度
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

  // --- 会话管理 ---
  const createNewThread = (title = "新对话", isDebug = false) => {
    const newId = uuidv4();
    const newThread = { id: newId, title: title, history: [], isDebug: isDebug };
    setThreads(prev => [newThread, ...prev]);
    setActiveThreadId(newId);
    setMessages([]);
    setIsDebugMode(isDebug);
    resetTyper();
  };

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
      setIsDebugMode(targetThread.isDebug || false);
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

    setMessages(prev => [...prev, { role: 'user', content: textToSend }]);
    setInput("");
    setIsLoading(true);
    resetTyper();
    setMessages(prev => [...prev, { role: 'ai', content: "" }]);
    abortControllerRef.current = new AbortController();

    // 如果是调试模式，可以自动附加前缀（后端支持后可移除）
    const finalQuery = isDebugMode ? `[调试模式] ${textToSend}` : textToSend;

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: finalQuery,
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

      setThreads(prev => prev.map(t =>
        t.id === activeThreadId && (t.title === "新对话" || t.title === "代码调试")
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

  // --- 语音逻辑 ---
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await sendAudioToBackend(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error("无法访问麦克风:", error);
      alert("无法访问麦克风，请检查权限设置。");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessingVoice(true);
    }
  };

  const sendAudioToBackend = async (audioBlob) => {
    const formData = new FormData();
    formData.append("file", audioBlob, "voice_input.webm");

    try {
      const response = await fetch(VOICE_API_URL, { method: "POST", body: formData });
      if (!response.ok) throw new Error("识别失败");
      const data = await response.json();
      if (data.text) setInput(prev => prev + data.text);
    } catch (error) {
      console.error("语音识别错误:", error);
      alert("语音识别失败，请重试");
    } finally {
      setIsProcessingVoice(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-50 text-gray-800 font-sans">
      {/* --- 侧边栏：四模块架构 --- */}
      <div className="w-64 bg-gray-900 text-white flex flex-col flex-shrink-0 shadow-xl z-20">

        {/* 标题 */}
        <div className="p-5 border-b border-gray-800">
          <h1 className="font-bold text-lg flex items-center gap-2 tracking-wide">
            <Bot className="text-blue-500" size={24} />
            工厂智能助手
          </h1>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar py-2">

          {/* 1. 培训助手 (Chat) */}
          <div className="mb-6 px-3">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 px-2 flex items-center gap-2">
              <GraduationCap size={14} /> 培训助手
            </h3>
            <button
              onClick={() => createNewThread("新对话", false)}
              disabled={isLoading}
              className="w-full flex items-center gap-2 bg-blue-600 hover:bg-blue-700 p-2.5 rounded-lg text-sm transition-all shadow-md mb-2 group"
            >
              <Plus size={16} className="group-hover:rotate-90 transition-transform" /> 开始新培训/提问
            </button>

            <div className="space-y-1 mt-2">
              {threads.filter(t => !t.isDebug).map(thread => (
                <button
                  key={thread.id}
                  onClick={() => switchThread(thread.id)}
                  disabled={isLoading}
                  className={`w-full text-left p-2 rounded-lg text-sm flex items-center gap-2 truncate transition-colors ${activeThreadId === thread.id ? 'bg-gray-800 text-white border-l-2 border-blue-500' : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
                    }`}
                >
                  <MessageSquare size={14} className="flex-shrink-0 opacity-70" />
                  <span className="truncate">{thread.title}</span>
                </button>
              ))}
            </div>
          </div>

          {/* 2. 采集助手 (Collection) */}
          <div className="mb-6 px-3">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 px-2 flex items-center gap-2">
              <Archive size={14} /> 采集助手
            </h3>
            <div className="space-y-1">
              <button
                onClick={() => setIsKbOpen(true)}
                className="w-full flex items-center gap-2 text-gray-400 hover:text-white hover:bg-gray-800 p-2 rounded-lg transition-colors text-sm"
              >
                <Database size={16} /> 知识库录入 (主动)
              </button>
              <button
                onClick={() => setIsUnansweredOpen(true)}
                className={`w-full flex items-center gap-2 p-2 rounded-lg transition-all text-sm group ${unansweredCount > 0 ? "text-orange-400 hover:text-orange-300 hover:bg-gray-800" : "text-gray-400 hover:text-white hover:bg-gray-800"
                  }`}
              >
                <ClipboardList size={16} className={unansweredCount > 0 ? "animate-pulse" : ""} />
                待解答归档 (被动)
                {unansweredCount > 0 && (
                  <span className="ml-auto bg-orange-500 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full">
                    {unansweredCount}
                  </span>
                )}
              </button>
            </div>
          </div>

          {/* 3. 生产监测 (Monitoring) */}
          <div className="mb-6 px-3">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 px-2 flex items-center gap-2">
              <BarChart2 size={14} /> 生产监测
            </h3>
            <button
              onClick={() => setIsDashboardOpen(true)}
              className="w-full flex items-center gap-2 text-emerald-400 hover:text-emerald-300 hover:bg-gray-800 p-2 rounded-lg transition-colors text-sm"
            >
              <BarChart2 size={16} />
              生命周期看板
            </button>
          </div>

          {/* 4. 调试助手 (Debug) */}
          <div className="mb-6 px-3">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 px-2 flex items-center gap-2">
              <Terminal size={14} /> 调试助手
            </h3>
            <button
              onClick={() => createNewThread("代码调试", true)}
              className={`w-full flex items-center gap-2 p-2 rounded-lg transition-colors text-sm ${isDebugMode ? 'bg-purple-900/50 text-purple-300' : 'text-purple-400 hover:text-purple-300 hover:bg-gray-800'
                }`}
            >
              <Bug size={16} />
              新建调试会话
            </button>
          </div>

        </div>

        {/* 底部信息 */}
        <div className="p-4 border-t border-gray-800 text-xs text-gray-500 text-center">
          V2.0.0 (Arch: 4-Mods)
        </div>
      </div>

      {/* --- 主界面区域 --- */}
      <div className="flex-1 flex flex-col relative bg-white">

        {/* 顶部状态栏 */}
        <div className="h-14 border-b flex items-center px-6 shadow-sm z-10 bg-white/80 backdrop-blur-md justify-between">
          <h1 className="font-semibold text-gray-700 flex items-center gap-2">
            {isDebugMode ? (
              <>
                <Terminal className="text-purple-600" size={20} />
                <span className="text-purple-900">调试助手模式</span>
                <span className="text-xs text-purple-500 font-normal border border-purple-200 px-2 py-0.5 rounded-full">Debug Mode</span>
              </>
            ) : (
              <>
                <GraduationCap className="text-blue-600" size={20} />
                <span>培训与知识助手</span>
              </>
            )}
          </h1>
          <div className="text-xs text-gray-400 flex items-center gap-1">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            系统在线
          </div>
        </div>

        {/* 聊天内容区 */}
        <div className="flex-1 overflow-y-auto p-4 pb-32 custom-scrollbar">
          <div className="max-w-3xl mx-auto space-y-6 min-h-full flex flex-col">

            {/* 欢迎界面 (根据模式切换) */}
            {messages.length === 0 && (
              <div className="flex-1 flex flex-col items-center justify-center text-center mt-10 animate-fade-in-up">
                <div className={`w-20 h-20 rounded-2xl shadow-sm border border-gray-100 flex items-center justify-center mb-6 ${isDebugMode ? 'bg-purple-50' : 'bg-blue-50'}`}>
                  {isDebugMode ? <Bug size={40} className="text-purple-600" /> : <Bot size={40} className="text-blue-600" />}
                </div>

                <h2 className="text-2xl font-bold text-gray-800 mb-3">
                  {isDebugMode ? "遇到代码或硬件报错了吗？" : "有什么可以帮你的吗？"}
                </h2>
                <p className="text-gray-500 mb-10 max-w-md">
                  {isDebugMode
                    ? "请输入详细的软硬件报错信息、日志片段，我将协助你进行代码调试和故障定位。"
                    : "我是您的全能工厂助手。我可以提供操作培训、查询图纸、或者记录您遇到的新问题。"
                  }
                </p>

                {/* 快捷卡片 */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl px-4">
                  {!isDebugMode ? (
                    // 培训模式的提示词
                    <>
                      <button onClick={() => handleSend("操作FANUC机器人时急停了怎么办")} className="welcome-card group">
                        <div className="icon-box bg-red-50 text-red-500 group-hover:bg-blue-50 group-hover:text-blue-600"><AlertTriangle size={20} /></div>
                        <div className="text-left">
                          <div className="title group-hover:text-blue-700">紧急故障处理 (培训)</div>
                          <div className="desc">操作FANUC机器人时急停了怎么办</div>
                        </div>
                      </button>
                      <button onClick={() => handleSend("教我使用自动分拣系统的手动操作页面")} className="welcome-card group">
                        <div className="icon-box bg-green-50 text-green-600 group-hover:bg-blue-50 group-hover:text-blue-600"><Wrench size={20} /></div>
                        <div className="text-left">
                          <div className="title group-hover:text-blue-700">操作规程教学 (培训)</div>
                          <div className="desc">教我使用自动分拣系统的手动操作页面</div>
                        </div>
                      </button>
                    </>
                  ) : (
                    // 调试模式的提示词
                    <>
                      <button onClick={() => handleSend("PLC连接超时，错误码 0x8002，请分析原因")} className="welcome-card group border-purple-200 hover:border-purple-400">
                        <div className="icon-box bg-purple-50 text-purple-600"><Terminal size={20} /></div>
                        <div className="text-left">
                          <div className="title text-purple-800">硬件通信调试</div>
                          <div className="desc">PLC连接超时，错误码 0x8002</div>
                        </div>
                      </button>
                      <button onClick={() => handleSend("Python脚本报 KeyError: 'status'，如何修复？")} className="welcome-card group border-purple-200 hover:border-purple-400">
                        <div className="icon-box bg-purple-50 text-purple-600"><Bug size={20} /></div>
                        <div className="text-left">
                          <div className="title text-purple-800">脚本异常修复</div>
                          <div className="desc">Python脚本报 KeyError: 'status'</div>
                        </div>
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* 消息流渲染 */}
            {messages.map((msg, idx) => {
              const isLastAiMessage = msg.role === 'ai' && idx === messages.length - 1;
              const contentToShow = isLastAiMessage && (isLoading || isTyping) ? displayedContent : msg.content;

              return (
                <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {msg.role === 'ai' && (
                    <div className={`w-8 h-8 rounded-full border flex items-center justify-center flex-shrink-0 mt-1 ${isDebugMode ? 'bg-purple-50 border-purple-100' : 'bg-blue-50 border-blue-100'}`}>
                      {isDebugMode ? <Bug size={16} className="text-purple-600" /> : <Bot size={16} className="text-blue-600" />}
                    </div>
                  )}

                  <div className={`max-w-[85%] p-4 rounded-2xl text-sm leading-7 shadow-sm ${msg.role === 'user'
                    ? (isDebugMode ? 'bg-purple-600 text-white rounded-br-none' : 'bg-blue-600 text-white rounded-br-none')
                    : 'bg-white border border-gray-100 text-gray-800 rounded-bl-none'
                    }`}>
                    {msg.role === 'ai' ? (
                      <div>
                        {(!contentToShow && isLoading) ? (
                          <div className="flex items-center gap-2 text-gray-400 py-1">
                            <Loader2 size={16} className="animate-spin" />
                            <span className="text-xs">
                              {isDebugMode ? "正在分析错误日志并检索解决方案..." : "正在检索知识库生成培训内容..."}
                            </span>
                          </div>
                        ) : (
                          <ReactMarkdown
                            components={{
                              // 样式保持不变，省略部分重复代码以保持简洁
                              ul: ({ node, ...props }) => <ul className="list-disc pl-4 my-2 space-y-1" {...props} />,
                              ol: ({ node, ...props }) => <ol className="list-decimal pl-4 my-2 space-y-1" {...props} />,
                              strong: ({ node, ...props }) => <span className={`font-bold px-1 rounded ${isDebugMode ? 'text-purple-700 bg-purple-50' : 'text-blue-700 bg-blue-50'}`} {...props} />,
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
                              img: ({ node, ...props }) => (
                                <img {...props} className="max-w-full h-auto rounded-lg shadow-md my-4 border border-gray-200 cursor-zoom-in" onClick={() => window.open(props.src, '_blank')} alt="示意图" />
                              )
                            }}
                          >
                            {contentToShow}
                          </ReactMarkdown>
                        )}
                        {isLastAiMessage && (isLoading || isTyping) && (
                          <span className={`inline-block w-1.5 h-4 ml-1 align-middle animate-pulse ${isDebugMode ? 'bg-purple-600' : 'bg-blue-600'}`}></span>
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

        {/* 底部输入框 */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pt-12 pb-6 px-4">
          <div className="max-w-3xl mx-auto relative group">

            {isRecording && (
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 bg-red-500 text-white px-4 py-1.5 rounded-full text-xs animate-pulse shadow-md flex items-center gap-2 z-20">
                <span className="w-2 h-2 bg-white rounded-full animate-ping"></span>
                正在录音... 点击麦克风结束
              </div>
            )}

            <div className={`bg-white border rounded-xl shadow-lg flex items-end p-2 gap-2 focus-within:ring-2 transition-all ${isDebugMode ? 'border-purple-200 focus-within:ring-purple-500/20 focus-within:border-purple-400' : 'border-gray-300 focus-within:ring-blue-500/20 focus-within:border-blue-400'
              }`}>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(null); }
                }}
                placeholder={isRecording ? "正在听你说话..." : (isDebugMode ? "粘贴报错代码或日志..." : "描述故障现象、询问操作步骤...")}
                className="w-full max-h-32 bg-transparent border-none focus:ring-0 resize-none p-3 text-gray-700 placeholder-gray-400 text-sm"
                rows={1}
                disabled={isLoading || isRecording || isProcessingVoice}
              />

              <div className="flex items-center mb-1 gap-1">
                {isProcessingVoice ? (
                  <div className="p-2 mr-1"><Loader2 size={20} className="animate-spin text-blue-500" /></div>
                ) : (
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={isLoading}
                    className={`p-2 rounded-lg transition-all mr-1 ${isRecording ? 'bg-red-100 text-red-600 animate-pulse' : 'bg-gray-100 text-gray-500 hover:bg-gray-200'}`}
                  >
                    {isRecording ? <StopCircle size={20} /> : <Mic size={20} />}
                  </button>
                )}

                {isLoading ? (
                  <button onClick={handleStop} className="p-2 rounded-lg bg-red-50 text-red-500 hover:bg-red-100"><StopCircle size={20} /></button>
                ) : (
                  <button
                    onClick={() => handleSend(null)}
                    disabled={!input.trim()}
                    className={`p-2 rounded-lg transition-all ${input.trim()
                      ? (isDebugMode ? 'bg-purple-600 text-white hover:bg-purple-700' : 'bg-blue-600 text-white hover:bg-blue-700')
                      : 'bg-gray-100 text-gray-300 cursor-not-allowed'
                      }`}
                  >
                    <Send size={18} />
                  </button>
                )}
              </div>
            </div>

            <p className="text-center text-xs text-gray-400 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
              {isDebugMode ? "调试模式下建议提供完整Log以便分析" : "内容由采集助手和培训助手协同生成"}
            </p>
          </div>
        </div>
      </div>

      {/* 弹窗组件保持不变 */}
      <KnowledgeModal isOpen={isKbOpen} onClose={() => setIsKbOpen(false)} />
      <UnansweredModal isOpen={isUnansweredOpen} onClose={() => setIsUnansweredOpen(false)} />
      <LifecycleDashboard isOpen={isDashboardOpen} onClose={() => setIsDashboardOpen(false)} />

      {/* 简单的 CSS 注入，用于一些 hover 效果 */}
      <style>{`
        .welcome-card {
            @apply flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left;
        }
        .welcome-card .icon-box {
            @apply w-10 h-10 rounded-lg flex items-center justify-center transition-colors;
        }
        .welcome-card .title {
            @apply font-semibold text-gray-700;
        }
        .welcome-card .desc {
            @apply text-xs text-gray-400;
        }
      `}</style>
    </div>
  );
}

export default App;