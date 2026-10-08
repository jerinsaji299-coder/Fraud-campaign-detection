import { Route, Routes } from 'react-router-dom'

import { Layout } from './components/Layout'
import { HomePage } from './pages/HomePage'
import { DatasetPage } from './pages/DatasetPage'
import { CampaignExplorerPage } from './pages/CampaignExplorerPage'
import { CampaignDetailPage } from './pages/CampaignDetailPage'
import { VisibilityLabPage } from './pages/VisibilityLabPage'
import { ResultsPage } from './pages/ResultsPage'
import { MethodologyPage } from './pages/MethodologyPage'
import { EmptyState } from './components/states'

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="dataset" element={<DatasetPage />} />
        <Route path="campaigns" element={<CampaignExplorerPage />} />
        <Route path="campaigns/:campaignId" element={<CampaignDetailPage />} />
        <Route path="visibility" element={<VisibilityLabPage />} />
        <Route path="results" element={<ResultsPage />} />
        <Route path="methodology" element={<MethodologyPage />} />
        <Route
          path="*"
          element={
            <EmptyState
              title="Page not found"
              description="That route does not exist."
            />
          }
        />
      </Route>
    </Routes>
  )
}
