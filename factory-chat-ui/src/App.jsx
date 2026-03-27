import { useState, useEffect } from 'react';
import {
    GraduationCap, Archive, BarChart2, Bug,
    Database, ClipboardList, ArrowLeft, Factory,
    ArrowRight, Activity, Server, ShieldCheck, Cpu, Sparkles,
    Video, MessageSquare, User, LogOut, Loader2, Images, Bot
} from 'lucide-react';
import TrainingAssistant from './TrainingAssistant';
import TrainingVideoManager from './TrainingVideoManager';
import DebugAssistant from './DebugAssistant';
import LifecycleDashboard from './components/LifecycleDashboard';
import KnowledgeModal from './components/KnowledgeModal';
import UnansweredModal from './components/UnansweredModal';
import ImageCollection from './components/ImageCollection';
import { API_BASE_URL } from './config';

export default function App() {
    const [currentModule, setCurrentModule] = useState('home');
    const [isKbOpen, setIsKbOpen] = useState(false);
    const [isUnansweredOpen, setIsUnansweredOpen] = useState(false);

    const [user, setUser] = useState(null);
    const [usernameInput, setUsernameInput] = useState("");
    const [passwordInput, setPasswordInput] = useState(""); // 新增密码状态
    const [loginError, setLoginError] = useState("");       // 新增错误提示
    const [isLoading, setIsLoading] = useState(false); // 新增：加载状态

    // 检查本地登录状态
    useEffect(() => {
        const savedUser = localStorage.getItem("factory_user");
        if (savedUser) {
            try {
                setUser(JSON.parse(savedUser));
            } catch (e) {
                localStorage.removeItem("factory_user");
            }
        }
    }, []);

    const handleLogin = async (e) => {
        if (e) e.preventDefault();

        if (!usernameInput.trim() || !passwordInput.trim()) {
            setLoginError("请输入用户名和密码");
            return;
        }

        setIsLoading(true);
        setLoginError("");

        try {
            const res = await fetch(`${API_BASE_URL}/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json' // 必须加上这个头，后端才知道是 JSON
                },
                body: JSON.stringify({
                    username: usernameInput,
                    password: passwordInput
                })
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({ detail: "登录失败" }));
                // 处理 FastAPI 可能返回的数组格式错误，也处理字符串格式
                const msg = Array.isArray(errData.detail)
                    ? errData.detail.map(e => `${e.loc.join('.')}: ${e.msg}`).join(", ")
                    : errData.detail;
                throw new Error(msg || `请求失败: ${res.status}`);
            }

            const data = await res.json();
            const userInfo = data.user || data;

            setUser(userInfo);
            localStorage.setItem("factory_user", JSON.stringify(userInfo));
            setLoginError("");
        } catch (err) {
            console.error("Login Error:", err);
            setLoginError(err.message || "无法连接到服务器");
        } finally {
            setIsLoading(false);
        }
    };
    const handleLogout = () => {
        setUser(null);
        localStorage.removeItem("factory_user");
        setCurrentModule('home');
        setPasswordInput(""); // 清空密码
    };

    // --- 登录 UI ---
    if (!user) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-slate-50">
                <div className="bg-white p-8 rounded-xl shadow-lg w-96 animate-fade-in-up">
                    <h2 className="text-2xl font-bold mb-6 text-center text-slate-800">🏭 智能分拣助手</h2>

                    <form onSubmit={handleLogin}> {/* 包裹在 form 中以支持回车提交 */}
                        <div className="mb-4">
                            <input
                                type="text"
                                placeholder="用户名"
                                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                                value={usernameInput}
                                onChange={(e) => setUsernameInput(e.target.value)}
                                disabled={isLoading}
                            />
                        </div>

                        <div className="mb-6">
                            <input
                                type="password"
                                placeholder="密码"
                                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                                value={passwordInput}
                                onChange={(e) => setPasswordInput(e.target.value)}
                                disabled={isLoading}
                            />
                        </div>

                        {loginError && (
                            <div className="mb-4 text-sm text-red-500 bg-red-50 p-2 rounded flex items-center gap-2">
                                <Bug size={14} /> {loginError}
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={isLoading}
                            className={`w-full p-3 rounded-lg font-medium transition-colors flex items-center justify-center gap-2
                                ${isLoading ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'} text-white`}
                        >
                            {isLoading ? <Loader2 className="animate-spin" size={20} /> : "安全登录"}
                        </button>
                    </form>

                    <p className="text-center text-xs text-gray-400 mt-4">
                        内部系统，请联系管理员获取账号
                    </p>
                </div>
            </div>
        );
    }
    // --- 1. 渲染主页 ---
    const renderHome = () => (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-indigo-50/40 relative overflow-hidden font-sans selection:bg-blue-200">
            {/* 简约背景装饰 */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#8882_1px,transparent_1px),linear-gradient(to_bottom,#8882_1px,transparent_1px)] bg-[size:64px_64px]"></div>

            {/* 柔和光晕 */}
            <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-200/40 rounded-full blur-[120px] -translate-y-1/4 translate-x-1/4"></div>
            <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-indigo-200/40 rounded-full blur-[120px] translate-y-1/4 -translate-x-1/4"></div>

            {/* 用户信息与退出按钮 */}
            <div className="absolute top-6 right-6 z-50 flex items-center gap-4 animate-fade-in">
                <div className="flex items-center gap-2 bg-white/60 backdrop-blur px-3 py-1.5 rounded-full border border-slate-200 shadow-sm">
                    <User size={16} className="text-slate-500" />
                    <span className="text-sm font-semibold text-slate-700">{user.username || "Admin"}</span>
                </div>
                <button
                    onClick={handleLogout}
                    className="flex items-center gap-2 px-3 py-1.5 bg-red-50 text-red-600 rounded-full hover:bg-red-100 transition-colors text-sm font-medium"
                >
                    <LogOut size={14} /> 退出
                </button>
            </div>

            <div className="relative z-10 flex flex-col items-center justify-center min-h-screen p-6 md:p-12">
                {/* 顶部标题区 */}
                <div className="text-center mb-20 animate-fade-in-down">
                    <div className="inline-flex items-center justify-center gap-3 px-6 py-3 mb-8 bg-white/80 backdrop-blur-sm border border-blue-100 rounded-full shadow-sm">
                        <Bot size={20} className="text-blue-500" />
                        <span className="text-sm font-semibold tracking-widest text-slate-600">华工小筑</span>
                        <Sparkles size={16} className="text-blue-500" />
                    </div>

                    <h1 className="text-6xl md:text-7xl font-black text-slate-800 mb-6 tracking-tight">
                        <span className="inline-block bg-clip-text text-transparent bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600">
                            智能分拣助手
                        </span>
                    </h1>
                    <p className="text-xl text-slate-600 max-w-3xl mx-auto leading-relaxed">
                        集成了自主决策、视觉感知与工业数据分析的下一代 AI Agent
                        <br className="hidden md:block" />
                        <span className="text-blue-600 font-medium">请选择您要进入的功能模块</span>
                    </p>
                </div>

                {/* 卡片网格区 */}
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 max-w-7xl w-full animate-fade-in-up mb-16">
                    {/* 1. 培训助手 */}
                    <ModuleCard
                        title="培训助手"
                        desc="交互式操作指导,查询设备手册、图纸与培训视频。"
                        icon={<GraduationCap size={28} />}
                        bgColor="from-blue-500 to-blue-600"
                        lightBg="bg-blue-50"
                        textColor="text-blue-600"
                        hoverShadow="hover:shadow-blue-200"
                        delay="0"
                        onClick={() => setCurrentModule('training-menu')}
                    />

                    {/* 2. 采集助手 */}
                    <ModuleCard
                        title="采集助手"
                        desc="主动录入知识文档与图片,或处理待解答问题。"
                        icon={<Archive size={28} />}
                        bgColor="from-orange-500 to-orange-600"
                        lightBg="bg-orange-50"
                        textColor="text-orange-600"
                        hoverShadow="hover:shadow-orange-200"
                        delay="100"
                        onClick={() => setCurrentModule('collection')}
                    />

                    {/* 3. 生产监测 */}
                    <ModuleCard
                        title="生产监测"
                        desc="生产日志可视化,零件全生命周期流转的精准追踪与多工序耗时瓶颈分析。"
                        icon={<BarChart2 size={28} />}
                        bgColor="from-emerald-500 to-emerald-600"
                        lightBg="bg-emerald-50"
                        textColor="text-emerald-600"
                        hoverShadow="hover:shadow-emerald-200"
                        delay="200"
                        onClick={() => setCurrentModule('monitoring')}
                    />

                    {/* 4. 调试助手 */}
                    <ModuleCard
                        title="调试助手"
                        desc="代码异常分析,机器故障联调与定位。"
                        icon={<Bug size={28} />}
                        bgColor="from-purple-500 to-purple-600"
                        lightBg="bg-purple-50"
                        textColor="text-purple-600"
                        hoverShadow="hover:shadow-purple-200"
                        delay="300"
                        onClick={() => setCurrentModule('debug')}
                    />
                </div>

                {/* 底部状态栏 */}
                <div className="w-full max-w-7xl border-t border-slate-200 pt-8 animate-fade-in">
                    <div className="flex flex-col md:flex-row justify-between items-center gap-6 text-slate-500 text-sm">
                        <div className="flex flex-wrap justify-center md:justify-start gap-4">
                            <div className="flex items-center gap-2 px-3 py-2 bg-white/60 rounded-lg border border-slate-200 shadow-sm">
                                <Server size={16} className="text-emerald-500" />
                                <span className="text-slate-600">Server: <span className="text-emerald-600 font-semibold">Stable</span></span>
                            </div>
                            <div className="flex items-center gap-2 px-3 py-2 bg-white/60 rounded-lg border border-slate-200 shadow-sm">
                                <Cpu size={16} className="text-blue-500" />
                                <span className="text-slate-600">AI Core: <span className="text-blue-600 font-semibold">Qwen3-VL-Plus</span></span>
                            </div>
                            <div className="flex items-center gap-2 px-3 py-2 bg-white/60 rounded-lg border border-slate-200 shadow-sm">
                                <Database size={16} className="text-purple-500" />
                                <span className="text-slate-600">RAG: <span className="text-purple-600 font-semibold">Elasticsearch</span></span>
                            </div>
                        </div>
                        <div className="text-slate-400 text-xs">© 2026 HGCYBER Data System Co., Ltd. All rights reserved.</div>
                    </div>
                </div>
            </div>
        </div>
    );

    // --- 2. 培训助手二级菜单 (新增) ---
    const renderTrainingMenu = () => (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-purple-50/40 relative overflow-hidden flex flex-col items-center justify-center p-8">
            {/* 背景效果 */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#8882_1px,transparent_1px),linear-gradient(to_bottom,#8882_1px,transparent_1px)] bg-[size:64px_64px]"></div>
            <div className="absolute top-1/4 right-1/4 w-[400px] h-[400px] bg-blue-200/50 rounded-full blur-[100px]"></div>
            <div className="absolute bottom-1/4 left-1/4 w-[400px] h-[400px] bg-purple-200/50 rounded-full blur-[100px]"></div>

            <button
                onClick={() => setCurrentModule('home')}
                className="absolute top-8 left-8 z-20 flex items-center gap-3 px-5 py-2.5 bg-white/80 backdrop-blur-sm rounded-full border border-slate-200 text-slate-600 hover:text-blue-600 hover:border-blue-300 hover:shadow-md transition-all group"
            >
                <ArrowLeft size={18} className="group-hover:-translate-x-1 transition-transform" />
                <span className="font-semibold">返回主页</span>
            </button>

            <div className="relative z-10 text-center mb-16 animate-fade-in-down">
                <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl mx-auto flex items-center justify-center mb-6 shadow-lg shadow-blue-200">
                    <GraduationCap size={36} className="text-white" />
                </div>
                <h2 className="text-5xl font-black text-slate-800 mb-4 tracking-tight">
                    <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">培训助手</span>
                </h2>
                <p className="text-slate-600 mt-4 text-lg">请选择您需要的培训服务类型</p>
            </div>

            <div className="relative z-10 grid grid-cols-1 md:grid-cols-2 gap-8 max-w-5xl w-full animate-fade-in-up">
                {/* 智能问答卡片 */}
                <button
                    onClick={() => setCurrentModule('training-chat')}
                    className="group relative flex flex-col items-center p-10 bg-white/80 backdrop-blur-sm rounded-3xl border border-slate-200 shadow-lg hover:shadow-xl hover:-translate-y-2 transition-all duration-300 text-center overflow-hidden"
                >
                    <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>

                    <div className="relative z-10 w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center text-white mb-6 shadow-lg shadow-blue-200 group-hover:scale-105 transition-transform">
                        <MessageSquare size={32} />
                    </div>

                    <h3 className="relative text-2xl font-bold text-slate-800 mb-3">智能问答</h3>
                    <p className="relative text-slate-600 leading-relaxed mb-8">
                        与 AI 助手对话，获取图文并茂的操作指导。
                        <br />
                        支持设备操作、故障排查等各类培训问题。
                    </p>

                    <div className="relative mt-auto flex items-center gap-2 text-blue-600 font-semibold group-hover:gap-3 transition-all">
                        <span>开始对话</span>
                        <ArrowRight size={18} />
                    </div>
                </button>

                {/* 视频库卡片 */}
                <button
                    onClick={() => setCurrentModule('training-video')}
                    className="group relative flex flex-col items-center p-10 bg-white/80 backdrop-blur-sm rounded-3xl border border-slate-200 shadow-lg hover:shadow-xl hover:-translate-y-2 transition-all duration-300 text-center overflow-hidden"
                >
                    <div className="absolute inset-0 bg-gradient-to-br from-purple-50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>

                    <div className="relative z-10 w-20 h-20 bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl flex items-center justify-center text-white mb-6 shadow-lg shadow-purple-200 group-hover:scale-105 transition-transform">
                        <Video size={32} />
                    </div>

                    <h3 className="relative text-2xl font-bold text-slate-800 mb-3">培训视频库</h3>
                    <p className="relative text-slate-600 leading-relaxed mb-8">
                        上传和查看培训视频资料。
                        <br />
                        支持在线播放、搜索和解析视频文件。
                    </p>

                    <div className="relative mt-auto flex items-center gap-2 text-purple-600 font-semibold group-hover:gap-3 transition-all">
                        <span>进入视频库</span>
                        <ArrowRight size={18} />
                    </div>
                </button>
            </div>
        </div>
    );

    // --- 3. 采集助手二级菜单 ---
    const renderCollectionMenu = () => (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-orange-50/30 to-amber-50/40 relative overflow-hidden flex flex-col items-center justify-center p-8">
            {/* 背景效果 */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#8882_1px,transparent_1px),linear-gradient(to_bottom,#8882_1px,transparent_1px)] bg-[size:64px_64px]"></div>
            <div className="absolute top-1/4 right-1/4 w-[400px] h-[400px] bg-orange-200/50 rounded-full blur-[100px]"></div>
            <div className="absolute bottom-1/4 left-1/4 w-[400px] h-[400px] bg-blue-200/50 rounded-full blur-[100px]"></div>

            <button
                onClick={() => setCurrentModule('home')}
                className="absolute top-8 left-8 z-20 flex items-center gap-3 px-5 py-2.5 bg-white/80 backdrop-blur-sm rounded-full border border-slate-200 text-slate-600 hover:text-blue-600 hover:border-blue-300 hover:shadow-md transition-all group"
            >
                <ArrowLeft size={18} className="group-hover:-translate-x-1 transition-transform" />
                <span className="font-semibold">返回主页</span>
            </button>

            <div className="relative z-10 text-center mb-16 animate-fade-in-down">
                <div className="w-20 h-20 bg-gradient-to-br from-orange-500 to-orange-600 rounded-2xl mx-auto flex items-center justify-center mb-6 shadow-lg shadow-orange-200">
                    <Archive size={36} className="text-white" />
                </div>
                <h2 className="text-5xl font-black text-slate-800 mb-4 tracking-tight">
                    <span className="bg-clip-text text-transparent bg-gradient-to-r from-orange-600 to-amber-600">数据采集</span>中心
                </h2>
                <p className="text-slate-600 mt-4 text-lg">请选择您要进行的数据操作类型</p>
            </div>

            <div className="relative z-10 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-7xl w-full animate-fade-in-up">
                {/* 主动收集卡片 */}
                <button
                    onClick={() => setIsKbOpen(true)}
                    className="group relative flex flex-col items-center p-10 bg-white/80 backdrop-blur-sm rounded-3xl border border-slate-200 shadow-lg hover:shadow-xl hover:-translate-y-2 transition-all duration-300 text-center overflow-hidden"
                >
                    <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>

                    <div className="relative z-10 w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center text-white mb-6 shadow-lg shadow-blue-200 group-hover:scale-105 transition-transform">
                        <Database size={32} />
                    </div>

                    <h3 className="relative text-2xl font-bold text-slate-800 mb-3">主动录入 (Knowledge)</h3>
                    <p className="relative text-slate-600 leading-relaxed mb-8">
                        上传 PDF 手册、Word 技术文档或 Excel 故障代码表。
                        <br />
                        系统将自动进行向量化处理并存入知识库。
                    </p>

                    <div className="relative mt-auto flex items-center gap-2 text-blue-600 font-semibold group-hover:gap-3 transition-all">
                        <span>开始录入</span>
                        <ArrowRight size={18} />
                    </div>
                </button>

                {/* 被动收集卡片 */}
                <button
                    onClick={() => setIsUnansweredOpen(true)}
                    className="group relative flex flex-col items-center p-10 bg-white/80 backdrop-blur-sm rounded-3xl border border-slate-200 shadow-lg hover:shadow-xl hover:-translate-y-2 transition-all duration-300 text-center overflow-hidden"
                >
                    <div className="absolute inset-0 bg-gradient-to-br from-orange-50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>

                    <div className="relative z-10 w-20 h-20 bg-gradient-to-br from-orange-500 to-orange-600 rounded-2xl flex items-center justify-center text-white mb-6 shadow-lg shadow-orange-200 group-hover:scale-105 transition-transform">
                        <ClipboardList size={32} />
                    </div>

                    <h3 className="relative text-2xl font-bold text-slate-800 mb-3">待解答归档 (Pending)</h3>
                    <p className="relative text-slate-600 leading-relaxed mb-8">
                        查看 AI 无法回答的用户提问记录。
                        <br />
                        通过人工补充答案,让知识库实现自我生长。
                    </p>

                    <div className="relative mt-auto flex items-center gap-2 text-orange-600 font-semibold group-hover:gap-3 transition-all">
                        <span>处理问题</span>
                        <ArrowRight size={18} />
                    </div>
                </button>

                {/* 图片收集卡片 */}
                <button
                    onClick={() => setCurrentModule('image_collection')}
                    className="group relative flex flex-col items-center p-10 bg-white/80 backdrop-blur-sm rounded-3xl border border-slate-200 shadow-lg hover:shadow-xl hover:-translate-y-2 transition-all duration-300 text-center overflow-hidden"
                >
                    {/* 悬停时的绿色渐变背景光晕 */}
                    <div className="absolute inset-0 bg-gradient-to-br from-emerald-50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>

                    {/* 图标容器 - 翠绿色渐变 */}
                    <div className="relative z-10 w-20 h-20 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-2xl flex items-center justify-center text-white mb-6 shadow-lg shadow-emerald-200 group-hover:scale-105 transition-transform">
                        <Images size={32} />
                    </div>

                    {/* 标题 */}
                    <h3 className="relative text-2xl font-bold text-slate-800 mb-3">图片采集库 (Gallery)</h3>

                    {/* 描述文本 */}
                    <p className="relative text-slate-600 leading-relaxed mb-8">
                        上传并标注工业现场图片，构建数据库。
                        <br />
                        积累视觉数据，让模型读懂复杂图纸与设备。
                    </p>

                    {/* 底部链接 - 绿色文本和箭头 */}
                    <div className="relative mt-auto flex items-center gap-2 text-emerald-600 font-semibold group-hover:gap-3 transition-all">
                        <span>进入图库</span>
                        <ArrowRight size={18} />
                    </div>
                </button>

            </div>

            {/* 挂载弹窗 */}
            <KnowledgeModal isOpen={isKbOpen} onClose={() => setIsKbOpen(false)} />
            <UnansweredModal isOpen={isUnansweredOpen} onClose={() => setIsUnansweredOpen(false)} />
        </div>
    );

    // --- 4. 路由分发 ---
    switch (currentModule) {
        case 'training-menu':
            return renderTrainingMenu();
        case 'training-chat':
            return <TrainingAssistant onBack={() => setCurrentModule('training-menu')} userId={user.user_id} />;
        case 'training-video':
            return <TrainingVideoManager onBack={() => setCurrentModule('training-menu')} />;
        case 'debug':
            return <DebugAssistant onBack={() => setCurrentModule('home')} userId={user.user_id} />;
        case 'monitoring':
            return <LifecycleDashboard isOpen={true} onClose={() => setCurrentModule('home')} />;
        case 'collection':
            return renderCollectionMenu();
        case 'image_collection':
            return <ImageCollection onBack={() => setCurrentModule('collection')} />;
        default:
            return renderHome();
    }
}

// --- 组件:简约卡片 ---
function ModuleCard({ title, desc, icon, bgColor, lightBg, textColor, hoverShadow, delay, onClick }) {
    return (
        <button
            onClick={onClick}
            style={{ animationDelay: `${delay}ms` }}
            className={`
                group relative flex flex-col h-full p-8
                bg-white/70 backdrop-blur-sm rounded-2xl 
                border border-slate-200 shadow-md ${hoverShadow}
                hover:shadow-xl hover:-translate-y-2 
                transition-all duration-300 text-left overflow-hidden
                animate-fade-in-up fill-mode-backwards
            `}
        >
            {/* 悬浮时的背景色 */}
            <div className={`absolute inset-0 ${lightBg} opacity-0 group-hover:opacity-100 transition-opacity duration-300`}></div>

            <div className="relative z-10 flex flex-col h-full">
                {/* 图标容器 */}
                <div className={`
                    w-14 h-14 rounded-xl mb-5 flex items-center justify-center text-white shadow-md
                    bg-gradient-to-br ${bgColor} group-hover:scale-110 transition-transform duration-300
                `}>
                    {icon}
                </div>

                <h3 className={`text-2xl font-bold text-slate-800 mb-3 group-hover:${textColor} transition-colors`}>
                    {title}
                </h3>

                <p className="text-slate-600 text-sm leading-relaxed mb-6 flex-grow">
                    {desc}
                </p>

                {/* 底部箭头 */}
                <div className="flex items-center justify-between border-t border-slate-200 pt-4 mt-auto">
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Enter Module</span>
                    <div className={`p-2 rounded-full bg-slate-100 text-slate-400 group-hover:text-white group-hover:bg-gradient-to-r ${bgColor} transition-all duration-300`}>
                        <ArrowRight size={16} className="group-hover:translate-x-0.5 transition-transform" />
                    </div>
                </div>
            </div>
        </button>
    );
}