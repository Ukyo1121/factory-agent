// src/components/UnansweredModal.jsx
import { useState, useEffect } from 'react';
import { ClipboardList, CheckCircle, Loader2, FileUp, FileText } from 'lucide-react'; // 引入 FileText 图标

export default function UnansweredModal({ isOpen, onClose }) {
    // --- 内部状态 ---
    const [unansweredList, setUnansweredList] = useState([]);
    const [selectedQuestion, setSelectedQuestion] = useState(null); // 当前选中的问题
    const [solveText, setSolveText] = useState(""); // 回答文本
    const [customFileName, setCustomFileName] = useState(""); // 自定义文件名
    const [solveFile, setSolveFile] = useState(null); // 回答文件
    const [isSolving, setIsSolving] = useState(false); // 提交Loading状态

    // --- 获取待解答列表 ---
    const fetchUnanswered = async () => {
        try {
            const res = await fetch("http://localhost:8000/admin/unanswered_questions");
            const data = await res.json();
            setUnansweredList(data.questions || []);
        } catch (e) {
            console.error("获取待解答列表失败:", e);
        }
    };

    // --- 监听打开状态 ---
    useEffect(() => {
        if (isOpen) {
            fetchUnanswered();
            // 重置状态
            setSelectedQuestion(null);
            setSolveText("");
            setCustomFileName("");
            setSolveFile(null);
        }
    }, [isOpen]);

    // --- 提交解答 ---
    const handleSolve = async () => {
        if (!solveText && !solveFile) return alert("请输入文字或上传文件");
        setIsSolving(true);

        const formData = new FormData();
        formData.append("query", selectedQuestion.query);

        if (solveText) {
            formData.append("answer_text", solveText);
            // 如果输入了文件名，则添加到请求中
            if (customFileName.trim()) {
                formData.append("custom_filename", customFileName.trim());
            }
        }

        if (solveFile) formData.append("file", solveFile);

        try {
            const res = await fetch("http://localhost:8000/admin/solve_question", {
                method: "POST",
                body: formData
            });
            if (!res.ok) throw new Error("Failed");

            alert("解答已提交，并成功入库！");
            // 成功后：重置表单，刷新列表，返回列表页
            setSolveText("");
            setCustomFileName("");
            setSolveFile(null);
            setSelectedQuestion(null);
            fetchUnanswered();
        } catch (e) {
            alert("提交失败: " + e.message);
        } finally {
            setIsSolving(false);
        }
    };

    // 如果未打开，不渲染任何内容
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4 backdrop-blur-sm">
            <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col animate-in fade-in zoom-in duration-200">

                {/* 顶部标题栏 */}
                <div className="p-4 border-b flex justify-between items-center bg-gray-50 rounded-t-xl">
                    <h3 className="font-bold text-gray-800 flex items-center gap-2">
                        <ClipboardList className="text-orange-500" size={20} />
                        待解答问题列表
                        {unansweredList.length > 0 && !selectedQuestion && (
                            <span className="bg-orange-100 text-orange-600 text-xs px-2 py-0.5 rounded-full border border-orange-200">
                                {unansweredList.length} 待处理
                            </span>
                        )}
                    </h3>
                    <button
                        onClick={onClose}
                        className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded-full transition-colors"
                    >
                        <span className="sr-only">关闭</span>
                        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* 内容区域 */}
                <div className="flex-1 overflow-y-auto p-4 custom-scrollbar bg-gray-50/50">
                    {selectedQuestion ? (
                        // === 界面 B: 填写解答 ===
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 space-y-4">
                            <div className="bg-orange-50 p-4 rounded-lg border border-orange-100">
                                <div className="text-xs text-orange-400 font-semibold mb-1 uppercase tracking-wide">待解决问题</div>
                                <div className="text-lg font-bold text-gray-800">{selectedQuestion.query}</div>
                                <div className="text-xs text-gray-500 mt-2 flex items-center gap-2">
                                    <span className="w-1.5 h-1.5 bg-red-400 rounded-full"></span>
                                    未检索原因: {selectedQuestion.reason}
                                </div>
                            </div>

                            <div className="pt-2 space-y-3">
                                <label className="block text-sm font-bold text-gray-700">人工解答 (输入文字)</label>

                                {/* 自定义文件名输入框 */}
                                <div className="flex items-center gap-2 mb-2">
                                    <FileText size={16} className="text-gray-400" />
                                    <input
                                        type="text"
                                        className="flex-1 border-b border-gray-200 focus:border-blue-500 outline-none text-sm py-1 bg-transparent placeholder-gray-300"
                                        placeholder="自定义生成的文件名 (可选，默认为随机ID)"
                                        value={customFileName}
                                        onChange={e => setCustomFileName(e.target.value)}
                                    />
                                    <span className="text-xs text-gray-400 font-mono">.txt</span>
                                </div>

                                <textarea
                                    className="w-full border border-gray-200 rounded-lg p-3 text-sm h-32 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none transition-all resize-none"
                                    placeholder="请输入详细的解决方案、操作步骤或维修建议..."
                                    value={solveText}
                                    onChange={e => setSolveText(e.target.value)}
                                />

                                <div className="relative flex py-2 items-center">
                                    <div className="flex-grow border-t border-gray-200"></div>
                                    <span className="flex-shrink-0 mx-4 text-gray-400 text-xs font-medium">或者</span>
                                    <div className="flex-grow border-t border-gray-200"></div>
                                </div>

                                <label className={`flex flex-col items-center justify-center w-full h-24 border-2 border-dashed rounded-lg cursor-pointer transition-all ${solveFile ? 'border-green-300 bg-green-50' : 'border-gray-300 hover:bg-gray-50 hover:border-blue-400'}`}>
                                    <div className="flex flex-col items-center justify-center pt-2 pb-3">
                                        {solveFile ? (
                                            <div className="flex items-center gap-2 text-green-700 font-medium animate-pulse">
                                                <CheckCircle size={24} />
                                                <span className="text-sm">{solveFile.name}</span>
                                            </div>
                                        ) : (
                                            <>
                                                <FileUp className="w-8 h-8 text-gray-400 mb-2" />
                                                <p className="text-xs text-gray-500">点击上传 PDF/Word 文档作为答案</p>
                                            </>
                                        )}
                                    </div>
                                    <input type="file" className="hidden" onChange={e => setSolveFile(e.target.files[0])} accept=".pdf,.docx,.doc,.txt" />
                                </label>
                            </div>

                            <div className="flex gap-3 pt-4 border-t mt-4">
                                <button
                                    onClick={() => {
                                        setSelectedQuestion(null);
                                        setSolveText("");
                                        setCustomFileName("");
                                        setSolveFile(null);
                                    }}
                                    className="flex-1 py-2.5 border border-gray-200 rounded-lg text-gray-600 hover:bg-gray-50 hover:text-gray-900 font-medium transition-colors"
                                >
                                    返回列表
                                </button>
                                <button
                                    onClick={handleSolve}
                                    disabled={isSolving}
                                    className="flex-1 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium shadow-sm transition-all active:scale-[0.98]"
                                >
                                    {isSolving ? <Loader2 className="animate-spin" size={18} /> : <CheckCircle size={18} />}
                                    提交并入库
                                </button>
                            </div>
                        </div>
                    ) : (
                        // === 界面 A: 问题列表 ===
                        <div className="space-y-3">
                            {unansweredList.length === 0 ? (
                                <div className="flex flex-col items-center justify-center py-16 text-center">
                                    <div className="w-16 h-16 bg-green-50 rounded-full flex items-center justify-center mb-4">
                                        <CheckCircle size={32} className="text-green-500" />
                                    </div>
                                    <h4 className="text-gray-800 font-medium">没有待解答的问题</h4>
                                    <p className="text-gray-400 text-sm mt-1">知识库非常完善🎉</p>
                                </div>
                            ) : (
                                unansweredList.map((q, idx) => (
                                    <div
                                        key={idx}
                                        className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 hover:border-blue-400 hover:shadow-md transition-all flex justify-between items-center group cursor-pointer"
                                        onClick={() => setSelectedQuestion(q)}
                                    >
                                        <div className="flex-1 min-w-0 pr-4">
                                            <div className="font-medium text-gray-800 line-clamp-1 group-hover:text-blue-700 transition-colors">
                                                {q.query}
                                            </div>
                                            <div className="text-xs text-gray-400 mt-1.5 flex items-center gap-2">
                                                <span className="bg-gray-100 px-1.5 py-0.5 rounded text-gray-500">{q.timestamp.split(' ')[0]}</span>
                                                <span>·</span>
                                                <span>{q.reason}</span>
                                            </div>
                                        </div>
                                        <button
                                            className="px-4 py-2 bg-blue-50 text-blue-600 text-sm font-medium rounded-lg group-hover:bg-blue-600 group-hover:text-white transition-colors flex-shrink-0"
                                        >
                                            解答
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}