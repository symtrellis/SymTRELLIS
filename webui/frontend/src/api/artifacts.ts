import type { ArtifactRef } from '../types';
import { postForm } from './client';

type UploadArtifactResponse = {
  artifact: ArtifactRef;
};

export function uploadInputImage(file: Blob | File, filename: string) {
  const formData = new FormData();
  formData.append('file', file, filename);
  formData.append('kind', 'input_image');

  return postForm<UploadArtifactResponse>('/api/artifacts/upload', formData);
}

export function artifactByRole(artifactRefs: Record<string, ArtifactRef>, roles: string[]) {
  return roles.reduce<ArtifactRef | null>(
    (artifact, role) => artifact ?? artifactRefs[role] ?? null,
    null,
  );
}
