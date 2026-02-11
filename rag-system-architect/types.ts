export interface FileNode {
  name: string;
  language: string;
  content: string;
}

export interface FolderNode {
  name: string;
  files: (FileNode | FolderNode)[];
}

export type FileSystem = (FileNode | FolderNode)[];

export enum Tab {
  ARCHITECTURE = 'ARCHITECTURE',
  SIMULATION = 'SIMULATION'
}