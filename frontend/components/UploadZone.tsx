"use client";

import React, { useCallback, useState, useRef } from "react";
import { Upload, Image as ImageIcon, Film, X } from "lucide-react";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  accept: string;
  mode: "image" | "video";
}

export default function UploadZone({
  onFileSelect,
  accept,
  mode,
}: UploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [fileSize, setFileSize] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleFile = useCallback(
    (file: File) => {
      setFileName(file.name);
      setFileSize(formatFileSize(file.size));

      if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target?.result as string);
        reader.readAsDataURL(file);
      } else {
        setPreview(null);
      }

      onFileSelect(file);
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleClick = () => fileInputRef.current?.click();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setPreview(null);
    setFileName(null);
    setFileSize(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const ModeIcon = mode === "image" ? ImageIcon : Film;

  return (
    <div
      id="upload-zone"
      className={`upload-zone relative cursor-pointer p-10 text-center transition-all duration-300 ${
        isDragOver ? "drag-over" : ""
      }`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={handleClick}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleInputChange}
        className="hidden"
        id="file-input"
      />

      {fileName ? (
        /* ── File selected state ── */
        <div className="animate-slide-up">
          <div className="flex flex-col items-center gap-4">
            {preview ? (
              <div className="relative w-48 h-32 rounded-lg overflow-hidden border border-white/10">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-full object-cover"
                />
              </div>
            ) : (
              <div className="w-20 h-20 rounded-xl bg-[#111a32] border border-white/10 flex items-center justify-center">
                <Film className="w-10 h-10 text-[#06b6d4]" />
              </div>
            )}
            <div>
              <p className="text-[#e8edf5] font-medium text-lg">{fileName}</p>
              <p className="text-[#8892a8] text-sm mt-1">{fileSize}</p>
            </div>
            <button
              onClick={clearFile}
              className="flex items-center gap-1.5 text-sm text-[#8892a8] hover:text-[#ef4444] transition-colors mt-2"
            >
              <X className="w-4 h-4" />
              Remove file
            </button>
          </div>
        </div>
      ) : (
        /* ── Empty state ── */
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 rounded-2xl bg-[#111a32] border border-white/10 flex items-center justify-center animate-glow-pulse">
            <Upload className="w-8 h-8 text-[#06b6d4]" />
          </div>
          <div>
            <p className="text-[#e8edf5] font-medium text-lg">
              Drop your {mode === "image" ? "image" : "video"} here
            </p>
            <p className="text-[#8892a8] text-sm mt-1">
              or click to browse files
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-[#4a5568]">
            <ModeIcon className="w-3.5 h-3.5" />
            <span>
              {mode === "image"
                ? "Supports JPG, PNG, BMP"
                : "Supports MP4, AVI, MOV, MKV"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
