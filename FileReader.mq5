//+------------------------------------------------------------------+
//|                                                LlenoReadFile.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

// Required counter to draw history. We only want to draw history in the first iteration of the tick
int loop_counter = 0;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   //Comment("Running Lleno prediction algo");
   string lowTF = "lowest_timeframe.txt";
   string lowname = "lowest_timeframe";
   string midname = "middle_timeframe";
   string highname = "highest_timeframe";
   string midTF = "middle_timeframe.txt";
   string highTF = "highest_timeframe.txt";
   bool foundlow = true;
   bool foundmid = true;
   bool foundhigh = true;
   const color lowcolor = clrDarkTurquoise;
   const color midcolor = clrTomato;
   const color highcolor = clrBlanchedAlmond;
   
   if(loop_counter == 0)
   {
      DrawHistory();
   }
   

// Get the bid price
   double Bid=NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID), _Digits);

   int spreadsheet = FileOpen(lowTF, FILE_READ|FILE_WRITE|FILE_ANSI, '\t', CP_ACP);
   if(spreadsheet != INVALID_HANDLE)
     {
      string pre_double = FileReadString(spreadsheet);
      double prediction = StringToDouble(pre_double);


      if(prediction > 0)
        {
         // Get the first visible candle on the chart.
         int CandlesOnChart=ChartGetInteger(0,CHART_FIRST_VISIBLE_BAR,0);

         //Darw line from prediction mate
         MqlRates PriceInformation[];

         // Organize array from the lowest candle
         ArraySetAsSeries(PriceInformation, true);

         int Data = CopyRates(_Symbol, _Period, 0, CandlesOnChart, PriceInformation);

         datetime NewTime = PriceInformation[0].time+PeriodSeconds(PERIOD_H1);
         //Comment("Predicition: "+PriceInformation[0].time);
         //Comment("2 days from now: "+(PriceInformation[0].time+(PeriodSeconds(PERIOD_D1)*2)));

         // Find Object
         int i = 0;
         while(foundlow==true)
           {
            int obj_id = ObjectFind(0,lowname+IntegerToString(i));

            if(obj_id >= 0)
              {
               i = i+1;
              }
            else
              {
               foundlow = false;
              }
           }
         string name_ = lowname+IntegerToString(i);
         ObjectCreate(_Symbol, name_,
                      OBJ_TREND, 0, PriceInformation[0].time,
                      PriceInformation[0].close, NewTime, prediction);

         ObjectSetInteger(0, name_, OBJPROP_COLOR,lowcolor);
        }

     } //my_time+= 23 * PeriodSeconds(PERIOD_H1);

   FileClose(spreadsheet);
   FileDelete(lowTF);
   
   int spreadsheet2 = FileOpen(midTF, FILE_READ|FILE_WRITE|FILE_ANSI, '\t', CP_ACP);
   if(spreadsheet2 != INVALID_HANDLE)
   {
      string pre_double = FileReadString(spreadsheet2);
      double prediction = StringToDouble(pre_double);

      if (prediction >0)
      {
         // Get the first visible candle on the chart.
         int CandlesOnChart=ChartGetInteger(0,CHART_FIRST_VISIBLE_BAR,0);

         //Darw line from prediction mate
         MqlRates PriceInformation[];

         // Organize array from the lowest candle
         ArraySetAsSeries(PriceInformation, true);

         int Data = CopyRates(_Symbol, _Period, 0, CandlesOnChart, PriceInformation);

         datetime NewTime = PriceInformation[0].time+PeriodSeconds(PERIOD_H4);

         // Find Object
         int i = 0;
         while(foundmid==true)
         {
            int obj_id = ObjectFind(0,midname+IntegerToString(i));

            if (obj_id >= 0)
            {
               i = i+1;
            }
            else
            {
            foundmid = false;
            }
         }
         string name_ = midname+IntegerToString(i);
         ObjectCreate(_Symbol, name_,
                      OBJ_TREND, 0, PriceInformation[0].time,
                      PriceInformation[0].close, NewTime, prediction);
         ObjectSetInteger(0, name_, OBJPROP_COLOR,midcolor);
      }
   } //my_time+= 23 * PeriodSeconds(PERIOD_H4);
   FileClose(spreadsheet2);
   FileDelete(midTF);
   int spreadsheet3 = FileOpen(highTF, FILE_READ|FILE_WRITE|FILE_ANSI, '\t', CP_ACP);
   if(spreadsheet3 != INVALID_HANDLE)
   {
      string pre_double = FileReadString(spreadsheet3);
      double prediction = StringToDouble(pre_double);


      if (prediction > 0)
      {
         // Get the first visible candle on the chart.
         int CandlesOnChart=ChartGetInteger(0,CHART_FIRST_VISIBLE_BAR,0);

         //Darw line from prediction mate
         MqlRates PriceInformation[];

         // Organize array from the lowest candle
         ArraySetAsSeries(PriceInformation, true);

         int Data = CopyRates(_Symbol, _Period, 0, CandlesOnChart, PriceInformation);

         datetime NewTime = PriceInformation[0].time+PeriodSeconds(PERIOD_D1);
         // Find Object
         int i = 0;
         while(foundhigh==true)
         {
            int obj_id = ObjectFind(0,highname+IntegerToString(i));

            if (obj_id >= 0)
            {
               i = i+1;
            }
            else
            {
            foundhigh = false;
            }
         }
         string name_ = highname+IntegerToString(i);
         ObjectCreate(_Symbol, name_,
                      OBJ_TREND, 0, PriceInformation[0].time,
                      PriceInformation[0].close, NewTime, prediction);
         ObjectSetInteger(0, name_, OBJPROP_COLOR,highcolor);

      }
   } //my_time+= 23 * PeriodSeconds(PERIOD_D1);
   FileClose(spreadsheet3);
   FileDelete(highTF);
   loop_counter = loop_counter+1;
  }
//+------------------------------------------------------------------+
//| Trade function                                                   |
//+------------------------------------------------------------------+
void OnTrade()
  {
//---

  }
//+------------------------------------------------------------------+


void DrawHistory()
{
   // Required variables
   bool foundlow = true;
   bool foundmid = true;
   bool foundhigh = true;
   string lowname = "lowest_timeframe";
   string midname = "middle_timeframe";
   string highname = "highest_timeframe";
   const color lowcolor = clrDarkTurquoise;
   const color midcolor = clrTomato;
   const color highcolor = clrBlanchedAlmond;
   string LowFile = "Historical data lowest_timeframe.txt";
   string MidFile = "Historical data middle_timeframe.txt";
   string HighFile = "Historical data highest_timeframe.txt";
   
   bool foundFrame[3];
   foundFrame[0] = foundlow;
   foundFrame[1] = foundmid;
   foundFrame[2] = foundhigh;
   
   color framecolors[3];
   framecolors[0] = lowcolor;   
   framecolors[1] = midcolor;
   framecolors[2] = highcolor;
   
   string files[3];
   files[0] = LowFile;
   files[1] = MidFile;
   files[2] = HighFile;
   
   string linenames[3];
   linenames[0] = lowname;
   linenames[1] = midname;
   linenames[2] = highname;

   // Get the first visible candle on the chart.
   int CandlesOnChart=ChartGetInteger(0,CHART_FIRST_VISIBLE_BAR,0);

   //Darw line from prediction mate
   MqlRates PriceInformation[];

   // Organize array from the lowest candle
   ArraySetAsSeries(PriceInformation, true);

   int Data = CopyRates(_Symbol, _Period, 0, CandlesOnChart, PriceInformation);
   
   
/**************************************************************************Lowest time frame*********************************************************************/

   int HistoryData = FileOpen(LowFile, FILE_READ|FILE_WRITE|FILE_ANSI, '\t', CP_ACP);
   if(HistoryData != INVALID_HANDLE)
   {
      while(!FileIsEnding(HistoryData)) 
      {
         string pre_double = FileReadString(HistoryData); 
     
         string sep=",";                // A separator as a character
         ushort u_sep;                  // The code of the separator character
         string result[];
         
         //--- Get the separator code
         u_sep=StringGetCharacter(sep,0);
         
         //--- Split the string to substrings
         int k=StringSplit(pre_double,u_sep,result);
         
         datetime datevar = StringToTime(result[0]);
         double prediction = StringToDouble(result[1]);
         
         int len = ArraySize(PriceInformation);

         int i = 0;
         int index = 0;
         for(i=0; i<len; i++)
         {
            

            if (TimeToString(PriceInformation[i].time) == TimeToString(datevar))
            {
               Comment("History: ", TimeToString(PriceInformation[i].time));
               
               if(prediction > 0)
                 {   
                  datetime NewTime = datevar+PeriodSeconds(PERIOD_H1);
                  
         
                  // Find Object
                  int j = 0;
                  while(foundlow==true)
                    {
                     int obj_id = ObjectFind(0,lowname+IntegerToString(j));
         
                     if(obj_id >= 0)
                       {
                        j = j+1;
                       }
                     else
                       {
                        foundlow = false;
                       }
                    }
                  foundlow = true;
                  string name_ = lowname+IntegerToString(j);
                  ObjectCreate(_Symbol, name_,
                               OBJ_TREND, 0, PriceInformation[i].time,
                               PriceInformation[i].close, NewTime, prediction);
                  
                  ObjectSetInteger(0, name_, OBJPROP_COLOR,lowcolor);
                  //Print("Plotting from "+TimeToString(PriceInformation[i].time)+" to "+TimeToString(NewTime));
                 }
               
            }
         }
      }     
   }
   FileClose(HistoryData);



/**************************************************************************Middle time frame*********************************************************************/   
int HistoryData2 = FileOpen(MidFile, FILE_READ|FILE_WRITE|FILE_ANSI, '\t', CP_ACP);
   if(HistoryData2 != INVALID_HANDLE)
   {
      while(!FileIsEnding(HistoryData2)) 
      {
         string pre_double = FileReadString(HistoryData); 
     
         string sep=",";                // A separator as a character
         ushort u_sep;                  // The code of the separator character
         string result[];
         
         //--- Get the separator code
         u_sep=StringGetCharacter(sep,0);
         
         //--- Split the string to substrings
         int k=StringSplit(pre_double,u_sep,result);
         
         datetime datevar = StringToTime(result[0]);
         double prediction = StringToDouble(result[1]);
         
         int len = ArraySize(PriceInformation);

         int i = 0;
         for(i=0; i<len; i++)
         {
            

            if (TimeToString(PriceInformation[i].time) == TimeToString(datevar))
            {
               Comment("History: ", TimeToString(PriceInformation[i].time));
              
               if(prediction > 0)
                 {   
                  
                  datetime NewTime = datevar+PeriodSeconds(PERIOD_H4);
                  
         
                  // Find Object
                  int j = 0;
                  while(foundmid==true)
                    {
                     int obj_id = ObjectFind(0,midname+IntegerToString(j));
         
                     if(obj_id >= 0)
                       {
                        j = j+1;
                       }
                     else
                       {
                        foundmid = false;
                       }
                    }
                  foundmid = true;
                  string name_ = midname+IntegerToString(j);
                  ObjectCreate(_Symbol, name_,
                               OBJ_TREND, 0, PriceInformation[i].time,
                               PriceInformation[i].close, NewTime, prediction);
                  
                  ObjectSetInteger(0, name_, OBJPROP_COLOR,midcolor);
                 }
               
            }
         }
      }     
   }
   FileClose(HistoryData2);

/**************************************************************************Highest time frame*********************************************************************/

   int HistoryData3 = FileOpen(HighFile, FILE_READ|FILE_WRITE|FILE_ANSI, '\t', CP_ACP);
   if(HistoryData3 != INVALID_HANDLE)
   {
      while(!FileIsEnding(HistoryData3)) 
      {
         string pre_double = FileReadString(HistoryData3); 
     
         string sep=",";                // A separator as a character
         ushort u_sep;                  // The code of the separator character
         string result[];
         
         //--- Get the separator code
         u_sep=StringGetCharacter(sep,0);
         
         //--- Split the string to substrings
         int k=StringSplit(pre_double,u_sep,result);
         
         datetime datevar = StringToTime(result[0]+" 00:00");
         double prediction = StringToDouble(result[1]);
         
         int len = ArraySize(PriceInformation);

         int i = 0;
         for(i=0; i<len; i++)
         {
            
            if (TimeToString(PriceInformation[i].time) == TimeToString(datevar))
            {
               Print("Price info: "+TimeToString(PriceInformation[i].time)+" Date var: "+TimeToString(datevar));
               Comment("History: ", TimeToString(PriceInformation[i].time));
              
               if(prediction > 0)
                 {   
                  
                  datetime NewTime = datevar+PeriodSeconds(PERIOD_D1);
                  
         
                  // Find Object
                  int j = 0;
                  while(foundhigh==true)
                    {
                     int obj_id = ObjectFind(0,highname+IntegerToString(j));
         
                     if(obj_id >= 0)
                       {
                        j = j+1;
                       }
                     else
                       {
                        foundhigh = false;
                       }
                    }
                  foundhigh = true;
                  string name_ = highname+IntegerToString(j);
                  ObjectCreate(_Symbol, name_,
                               OBJ_TREND, 0, PriceInformation[i].time,
                               PriceInformation[i].close, NewTime, prediction);
                  
                  ObjectSetInteger(0, name_, OBJPROP_COLOR,highcolor);
                 }
               
            }
         }
      }     
   }
   FileClose(HistoryData3);
}