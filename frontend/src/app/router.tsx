import { createBrowserRouter, Navigate } from 'react-router';
import { MobileLayout } from '@/components/MobileLayout';
import { WebLayout } from '@/components/WebLayout';
import { ROUTES } from '@/constants/routes';
import { AnalyzingPage } from '@/pages/Analyzing';
import { ExerciseSelectPage } from '@/pages/ExerciseSelect';
import { ResultPage } from '@/pages/Result';
import { UploadPage } from '@/pages/Upload';

export const router = createBrowserRouter([
  {
    element: <MobileLayout />,
    children: [
      { path: ROUTES.home, element: <ExerciseSelectPage /> },
      { path: ROUTES.upload, element: <UploadPage /> },
      { path: ROUTES.analyzing, element: <AnalyzingPage /> },
      { path: '*', element: <Navigate to={ROUTES.home} replace /> },
    ],
  },
  {
    // 결과 화면만 넓은 화면 기준 (2026-11-26 결정)
    element: <WebLayout />,
    children: [{ path: ROUTES.result, element: <ResultPage /> }],
  },
]);
