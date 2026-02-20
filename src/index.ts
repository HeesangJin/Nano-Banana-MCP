#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequest,
  CallToolRequestSchema,
  CallToolResult,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import { GoogleGenAI } from "@google/genai";
import { z } from "zod";
import fs from "fs/promises";
import path from "path";
import os from "os";
import { config as dotenvConfig } from "dotenv";

dotenvConfig();

const DEFAULT_IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation";
const LEGACY_IMAGE_MODEL = "gemini-2.5-flash-image-preview";

const ConfigSchema = z.object({
  geminiApiKey: z.string().min(1, "Gemini API key is required"),
});

type Config = z.infer<typeof ConfigSchema>;

type ConfigSource = "environment" | "config_file" | "not_configured";

class NanoBananaMCP {
  private server: Server;
  private genAI: GoogleGenAI | null = null;
  private config: Config | null = null;
  private configSource: ConfigSource = "not_configured";
  private lastImagePath: string | null = null;

  constructor() {
    this.server = new Server(
      {
        name: "nano-banana-mcp",
        version: "1.0.0",
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupHandlers();
  }

  private setupHandlers(): void {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: "configure_gemini_token",
          description: "Configure your Gemini API token for nano-banana image generation",
          inputSchema: {
            type: "object",
            properties: {
              apiKey: {
                type: "string",
                description: "Your Gemini API key from Google AI Studio",
              },
            },
            required: ["apiKey"],
          },
        },
        {
          name: "generate_image",
          description: "Generate a NEW image from text prompt. Returns only saved file path.",
          inputSchema: {
            type: "object",
            properties: {
              prompt: {
                type: "string",
                description: "Text prompt describing the NEW image to create from scratch",
              },
            },
            required: ["prompt"],
          },
        },
        {
          name: "edit_image",
          description: "Edit a SPECIFIC existing image file and return only saved file path.",
          inputSchema: {
            type: "object",
            properties: {
              imagePath: {
                type: "string",
                description: "Full file path to the main image file to edit",
              },
              prompt: {
                type: "string",
                description: "Text describing the modifications to make to the existing image",
              },
              referenceImages: {
                type: "array",
                items: { type: "string" },
                description: "Optional array of file paths to additional reference images",
              },
            },
            required: ["imagePath", "prompt"],
          },
        },
        {
          name: "get_configuration_status",
          description: "Check if Gemini API token is configured",
          inputSchema: {
            type: "object",
            properties: {},
            additionalProperties: false,
          },
        },
        {
          name: "continue_editing",
          description: "Continue editing the LAST image from this session and return only saved file path.",
          inputSchema: {
            type: "object",
            properties: {
              prompt: {
                type: "string",
                description: "Text describing modifications to make to the last image",
              },
              referenceImages: {
                type: "array",
                items: { type: "string" },
                description: "Optional array of file paths to additional reference images",
              },
            },
            required: ["prompt"],
          },
        },
        {
          name: "get_last_image_info",
          description: "Get information about the last generated/edited image in this session",
          inputSchema: {
            type: "object",
            properties: {},
            additionalProperties: false,
          },
        },
      ] as Tool[],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request: CallToolRequest): Promise<CallToolResult> => {
      try {
        switch (request.params.name) {
          case "configure_gemini_token":
            return await this.configureGeminiToken(request);
          case "generate_image":
            return await this.generateImage(request);
          case "edit_image":
            return await this.editImage(request);
          case "continue_editing":
            return await this.continueEditing(request);
          case "get_last_image_info":
            return await this.getLastImageInfo();
          case "get_configuration_status":
            return await this.getConfigurationStatus();
          default:
            throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${request.params.name}`);
        }
      } catch (error) {
        if (error instanceof McpError) {
          throw error;
        }
        throw new McpError(
          ErrorCode.InternalError,
          `Tool execution failed: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    });
  }

  private textResult(text: string): CallToolResult {
    return {
      content: [
        {
          type: "text",
          text,
        },
      ],
    };
  }

  private async configureGeminiToken(request: CallToolRequest): Promise<CallToolResult> {
    const { apiKey } = request.params.arguments as { apiKey: string };

    try {
      this.config = ConfigSchema.parse({ geminiApiKey: apiKey });
      this.genAI = new GoogleGenAI({ apiKey: this.config.geminiApiKey });
      this.configSource = "config_file";
      await this.saveConfig();
      return this.textResult("configured");
    } catch (error) {
      if (error instanceof z.ZodError) {
        throw new McpError(ErrorCode.InvalidParams, `Invalid API key: ${error.errors[0]?.message}`);
      }
      throw error;
    }
  }

  private async generateImage(request: CallToolRequest): Promise<CallToolResult> {
    this.requireConfigured();

    const { prompt } = request.params.arguments as { prompt: string };
    const { response } = await this.generateContentWithModelFallback(prompt);

    const imagesDir = this.getImagesDirectory();
    await fs.mkdir(imagesDir, { recursive: true, mode: 0o755 });

    const saved = await this.extractAndSaveImages(response, imagesDir, "generated");
    if (saved.length === 0) {
      throw new McpError(ErrorCode.InternalError, "No image data was returned by Gemini.");
    }

    return this.textResult(saved[0]);
  }

  private async editImage(request: CallToolRequest): Promise<CallToolResult> {
    this.requireConfigured();

    const { imagePath, prompt, referenceImages } = request.params.arguments as {
      imagePath: string;
      prompt: string;
      referenceImages?: string[];
    };

    const mainBuffer = await fs.readFile(imagePath);
    const imageParts: Array<{ inlineData?: { data: string; mimeType: string }; text?: string }> = [
      {
        inlineData: {
          data: mainBuffer.toString("base64"),
          mimeType: this.getMimeType(imagePath),
        },
      },
    ];

    if (referenceImages && referenceImages.length > 0) {
      for (const refPath of referenceImages) {
        try {
          const refBuffer = await fs.readFile(refPath);
          imageParts.push({
            inlineData: {
              data: refBuffer.toString("base64"),
              mimeType: this.getMimeType(refPath),
            },
          });
        } catch {
          // Skip unreadable reference images.
        }
      }
    }

    imageParts.push({ text: prompt });

    const { response } = await this.generateContentWithModelFallback([
      {
        parts: imageParts,
      },
    ]);

    const imagesDir = this.getImagesDirectory();
    await fs.mkdir(imagesDir, { recursive: true, mode: 0o755 });

    const saved = await this.extractAndSaveImages(response, imagesDir, "edited");
    if (saved.length === 0) {
      throw new McpError(ErrorCode.InternalError, "No edited image data was returned by Gemini.");
    }

    return this.textResult(saved[0]);
  }

  private async continueEditing(request: CallToolRequest): Promise<CallToolResult> {
    this.requireConfigured();

    if (!this.lastImagePath) {
      throw new McpError(
        ErrorCode.InvalidRequest,
        "No previous image found. Please generate or edit an image first."
      );
    }

    try {
      await fs.access(this.lastImagePath);
    } catch {
      throw new McpError(
        ErrorCode.InvalidRequest,
        `Last image file not found at: ${this.lastImagePath}. Please generate a new image first.`
      );
    }

    const { prompt, referenceImages } = request.params.arguments as {
      prompt: string;
      referenceImages?: string[];
    };

    return this.editImage({
      method: "tools/call",
      params: {
        name: "edit_image",
        arguments: {
          imagePath: this.lastImagePath,
          prompt,
          referenceImages,
        },
      },
    } as CallToolRequest);
  }

  private async getLastImageInfo(): Promise<CallToolResult> {
    if (!this.lastImagePath) {
      return this.textResult("none");
    }

    try {
      const stats = await fs.stat(this.lastImagePath);
      return this.textResult(
        JSON.stringify({
          path: this.lastImagePath,
          sizeBytes: stats.size,
          modifiedAt: stats.mtime.toISOString(),
        })
      );
    } catch {
      return this.textResult(JSON.stringify({ path: this.lastImagePath, exists: false }));
    }
  }

  private async getConfigurationStatus(): Promise<CallToolResult> {
    const configured = this.config !== null && this.genAI !== null;
    return this.textResult(
      JSON.stringify({
        configured,
        source: this.configSource,
      })
    );
  }

  private requireConfigured(): void {
    if (!this.config || !this.genAI) {
      throw new McpError(ErrorCode.InvalidRequest, "Gemini API token not configured. Use configure_gemini_token first.");
    }
  }

  private getImageModelCandidates(): string[] {
    const configuredModel = process.env.GEMINI_IMAGE_MODEL?.trim();
    const candidates = [configuredModel, DEFAULT_IMAGE_MODEL, LEGACY_IMAGE_MODEL].filter(
      (model): model is string => Boolean(model && model.length > 0)
    );
    return [...new Set(candidates)];
  }

  private isModelNotAvailableError(error: unknown): boolean {
    const message = error instanceof Error ? error.message : String(error);
    return (
      message.includes("NOT_FOUND") ||
      message.includes("is not found") ||
      message.includes("not supported for generateContent")
    );
  }

  private async generateContentWithModelFallback(contents: any): Promise<{ response: any; model: string }> {
    const models = this.getImageModelCandidates();
    let lastError: unknown = null;

    for (const model of models) {
      try {
        const response = await this.genAI!.models.generateContent({
          model,
          contents,
        });
        return { response, model };
      } catch (error) {
        lastError = error;
        if (!this.isModelNotAvailableError(error)) {
          throw error;
        }
      }
    }

    const details = lastError instanceof Error ? lastError.message : String(lastError);
    throw new McpError(
      ErrorCode.InternalError,
      `No compatible Gemini image model found. Tried: ${models.join(", ")}. Last error: ${details}`
    );
  }

  private async extractAndSaveImages(response: any, imagesDir: string, prefix: "generated" | "edited"): Promise<string[]> {
    const savedFiles: string[] = [];
    const parts = response?.candidates?.[0]?.content?.parts;

    if (!Array.isArray(parts)) {
      return savedFiles;
    }

    for (const part of parts) {
      const data = part?.inlineData?.data;
      if (!data) {
        continue;
      }

      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const randomId = Math.random().toString(36).substring(2, 8);
      const fileName = `${prefix}-${timestamp}-${randomId}.png`;
      const filePath = path.join(imagesDir, fileName);

      await fs.writeFile(filePath, Buffer.from(data, "base64"));
      this.lastImagePath = filePath;
      savedFiles.push(filePath);
    }

    return savedFiles;
  }

  private getMimeType(filePath: string): string {
    const ext = path.extname(filePath).toLowerCase();
    switch (ext) {
      case ".jpg":
      case ".jpeg":
        return "image/jpeg";
      case ".png":
        return "image/png";
      case ".webp":
        return "image/webp";
      default:
        return "image/jpeg";
    }
  }

  private getImagesDirectory(): string {
    const platform = os.platform();
    if (platform === "win32") {
      return path.join(os.homedir(), "Documents", "nano-banana-images");
    }

    const cwd = process.cwd();
    if (cwd.startsWith("/usr/") || cwd.startsWith("/opt/") || cwd.startsWith("/var/")) {
      return path.join(os.homedir(), "nano-banana-images");
    }

    return path.join(cwd, "generated_imgs");
  }

  private async saveConfig(): Promise<void> {
    if (!this.config) {
      return;
    }
    const configPath = path.join(process.cwd(), ".nano-banana-config.json");
    await fs.writeFile(configPath, JSON.stringify(this.config, null, 2));
  }

  private async loadConfig(): Promise<void> {
    const envApiKey = process.env.GEMINI_API_KEY;
    if (envApiKey) {
      try {
        this.config = ConfigSchema.parse({ geminiApiKey: envApiKey });
        this.genAI = new GoogleGenAI({ apiKey: this.config.geminiApiKey });
        this.configSource = "environment";
        return;
      } catch {
        // Ignore invalid environment key and fall back to file.
      }
    }

    try {
      const configPath = path.join(process.cwd(), ".nano-banana-config.json");
      const configData = await fs.readFile(configPath, "utf-8");
      this.config = ConfigSchema.parse(JSON.parse(configData));
      this.genAI = new GoogleGenAI({ apiKey: this.config.geminiApiKey });
      this.configSource = "config_file";
    } catch {
      this.configSource = "not_configured";
    }
  }

  public async run(): Promise<void> {
    await this.loadConfig();
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
  }
}

const server = new NanoBananaMCP();
server.run().catch(console.error);
